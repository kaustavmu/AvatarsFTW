# Copyright (c) SenseTime Research. All rights reserved.


import os
import argparse 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.ImagesDataset import ImagesDataset
import controlnet_aux

import cv2
import time
import copy
import imutils
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from tqdm import tqdm, trange
from xgboost import XGBRegressor, XGBClassifier

# for openpose body keypoint detector : # (src:https://github.com/Hzzone/pytorch-openpose)
from openpose.src import util
from openpose.src.body import Body

import math

from ultralytics import YOLO

import logging
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import shutil
import hydra
import yaml
from omegaconf import OmegaConf                                         
from torch.utils.data._utils.collate import default_collate                                             
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

from random import choice
from string import ascii_uppercase
from pti.pti_configs import global_config, paths_config
import wandb

from pti.training.coaches.multi_id_coach import MultiIDCoach
from pti.training.coaches.single_id_coach import SingleIDCoach

LOGGER = logging.getLogger(__name__)

def impute_with_xgboost(data):
    data = data.T
    data = np.where(data == -1, np.nan, data)
    data = pd.DataFrame(data)
    data_copy = data.copy()

    for column in data_copy.columns:
        if data[column].isnull().sum() > 0:
            
            non_missing = data_copy.loc[data[column].notna()]
            missing = data_copy.loc[data[column].isna()]
            
            X_train = non_missing.drop(columns=[column])
            y_train = non_missing[column]
            X_missing = missing.drop(columns=[column])
            
            if data[column].dtype == np.float64 or data[column].dtype == np.int64:
                model = XGBRegressor(n_estimators=100, random_state=42)
            else:
                model = XGBClassifier(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_missing)
            data_copy.loc[data[column].isna(), column] = predictions
    
    return data_copy


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def run_PTI(img_path, mask_path):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    global_config.device = "cpu"

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    # print('embedding_dir_path: ', embedding_dir_path) #./embeddings/barcelona/PTI
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(img_path, transforms.Compose([
        transforms.Resize((1024, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    mask = cv2.imread(mask_path)
    mask = np.transpose(mask, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    coach = SingleIDCoach(dataloader, mask, use_wandb)
    coach.train()

    return global_config.run_name

def run(args):
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    samples = np.load("samples.npy")
    sample_average = np.mean(samples, axis=0)

    #dataset = ImagesDataset(args.image_folder, transforms.Compose([transforms.ToTensor()]))
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    body_estimation = Body('openpose/model/body_pose_model.pth')
    yolo_model = YOLO("yolo11n-seg.pt")

    #total = len(dataloader)
    #print('Num of dataloader : ', total)
    os.makedirs(f'{args.output_folder}/middle_result', exist_ok=True)
    
    for image_name in tqdm(os.listdir(args.image_folder)):
        # try:
        ## tensor to numpy image
        print(f'Processing \'{image_name}\'')
        
        image = cv2.imread(args.image_folder + image_name)

        results = yolo_model([args.image_folder + image_name], device='cpu')
        result = results[0]
        
        boxes = result.boxes
        masks = result.masks
 
        b_mask = np.zeros(image.shape[:2], np.uint8)
        box_mask = np.zeros(image.shape[:2], np.uint8)

        human_idx = np.where(boxes.cls == 0)[0]
        object_idx = np.where(boxes.cls != 0)[0]

        xmin, ymin, xmax, ymax = np.int32(boxes.xyxy[human_idx[0]]) 

        if len(human_idx) != 1:
            print("More or less than one human detected.")
            continue

        contour = masks.xy[human_idx[0]]
        contour = contour.astype(np.int32)
        contour = contour.reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        box_mask[ymin:ymax, xmin:xmax] = 1

        full_obj_mask = np.zeros(image.shape[:2], np.uint8)

        for idx in object_idx:
            
            obj_mask = np.zeros(image.shape[:2], np.uint8)
            obj_contour = masks.xy[idx]
            obj_contour = obj_contour.astype(np.int32)
            obj_contour = obj_contour.reshape(-1, 1, 2)
            _ = cv2.drawContours(obj_mask, [obj_contour], -1, (1, 1, 1), cv2.FILLED)
            
            combined_mask = obj_mask + box_mask

            full_obj_mask[combined_mask == 2] = 255

        seg_res = cv2.bitwise_and(image, image, mask = b_mask)
        inv_mask = cv2.bitwise_not(b_mask)
        seg_res[np.ma.make_mask(inv_mask)] = [255, 255, 255]
 
        keypoints, subset = body_estimation(image)

        plt.imshow(seg_res[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()

        canvas = copy.deepcopy(image)
        canvas = util.draw_bodypose(canvas, keypoints, subset)
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()
        
        old_subset = copy.deepcopy(subset)
        subset = np.int8(subset[0][:-2])

        # Body part defaults from analysis of 100 samples
        top_Y = 0.005839198711285836
        shoulder_Y = 0.15934437235425183
        hip_Y = 0.4879426030016212
        knee_Y = 0.7456291434545443
        feet_Y = 0.9829986933151402
        L_shoulder_X = 0.17877904094024885
        R_shoulder_X = 0.8206836530646126
        L_knee_X = 0.3149378362263144
        R_knee_X = 0.7005089581371317

        min_X = min(keypoints[:, 0])
        max_X = max(keypoints[:, 0])
        min_Y = min(keypoints[:, 1])
        max_Y = max(keypoints[:, 1])

        min_Y_ratio = 1
        max_Y_ratio = 0

        # find min and max Y ratios
        if subset[14] != -1 or subset[15] != -1:
            min_Y_ratio = min(min_Y_ratio, top_Y)
            max_Y_ratio = max(max_Y_ratio, top_Y)
        if subset[2] != -1 or subset[5] != -1:
            min_Y_ratio = min(min_Y_ratio, shoulder_Y)
            max_Y_ratio = max(max_Y_ratio, shoulder_Y)
        if subset[8] != -1 or subset[11] != -1:
            min_Y_ratio = min(min_Y_ratio, hip_Y)
            max_Y_ratio = max(max_Y_ratio, hip_Y)
        if subset[9] != -1 or subset[12] != -1:
            min_Y_ratio = min(min_Y_ratio, knee_Y)
            max_Y_ratio = max(max_Y_ratio, knee_Y)
        if subset[10] != -1 or subset[13] != -1:
            min_Y_ratio = min(min_Y_ratio, feet_Y)
            max_Y_ratio = max(max_Y_ratio, feet_Y)

        # find min and max X
        if subset[2] != -1 or subset[5] != -1:
            if subset[2] != -1 and subset[5] != -1:
                r = keypoints[subset[5]][0]
                l = keypoints[subset[2]][0]
                gap = r - l
                total = gap/(R_shoulder_X - L_shoulder_X)
                min_X = min(min_X, l - total * L_shoulder_X)
                max_X = max(max_X, r + total * (1-R_shoulder_X))
            elif subset[2] != -1:
                max_X = max(max_X, min_X + (keypoints[subset[2]][0] - min_X)/L_shoulder_X)
            elif subset[5] != -1:
                min_X = min(min_X, max_X - (max_X - keypoints[subset[5]][0])/(1-R_shoulder_X))
        elif subset[9] != -1 or subset[12] != -1:
            if subset[9] != -1 and subset[12] != -1:
                r = keypoints[subset[12]][0]
                l = keypoints[subset[9]][0]
                gap = r - l
                total = gap/(R_knee_X - L_knee_X)
                min_X = min(min_X, l - total * L_knee_X)
                max_X = max(max_X, r + total * (1-R_knee_X))
            elif subset[9] != -1:
                max_X = max(max_X, min_X + (keypoints[subset[9]][0] - min_X)/L_knee_X)
            elif subset[12] != -1:
                min_X = min(min_X, max_X - (max_X - keypoints[subset[12]][0])/(1-R_knee_X))
        else:
            print("Too little information to align. Skipping image.")
            continue

        t = (max_Y - min_Y)/(max_Y_ratio - min_Y_ratio)
        min_Y -= t * min_Y_ratio
        max_Y += t * (1 - max_Y_ratio)

        w = max_X - min_X
        h = max_Y - min_Y

        res = []

        for i in range(len(subset)):
            if subset[i] == -1:
                res.append(-1)
                res.append(-1)
            else:
                x, y = keypoints[subset[i]][:2]
                x = (x-min_X)/w
                y = (y-min_Y)/h
                res.append(x)
                res.append(y)

        res = np.array([res])

        data = np.concatenate((samples, res))
        imputed_data = impute_with_xgboost(data)

        new_data = imputed_data.to_numpy()
        res = new_data.T[-1]

        diff = np.linalg.norm(res - sample_average)

        if diff < 1:

            new_keypoints = []
            new_subset = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 18]])

            for i in range(18):
                new_keypoints.append([int(min_X + w*res[2*i]), int(min_Y + h*res[2*i+1]), 1., i])
            
            new_keypoints = np.array(new_keypoints)

        else:
            
            new_keypoints = keypoints
            new_subset = old_subset
        
        canvas = np.ones((max(int(max_Y), ymax) + 100, min(int(max_X), xmax) + 100, 3))  
        final_mask = np.ones(canvas.shape)
        final_obj_mask = np.ones(canvas.shape)

        canvas[ymin:ymax, xmin:xmax] = seg_res[ymin:ymax, xmin:xmax]/255
        final_mask[ymin:ymax, xmin:xmax] = np.dstack([inv_mask, inv_mask, inv_mask])[ymin:ymax, xmin:xmax]/255
        final_obj_mask[ymin:ymax, xmin:xmax] = np.dstack([full_obj_mask, full_obj_mask, full_obj_mask])[ymin:ymax, xmin:xmax]/255


        f_min_y = np.int32(np.rint(min(ymin, min_Y)))
        f_max_y = np.int32(np.rint(max(ymax, max_Y)))
        f_min_x = np.int32(np.rint(min(xmin, min_X)))
        f_max_x = np.int32(np.rint(max(xmax, max_X)))

        new_keypoints[:, 0] -= f_min_x
        new_keypoints[:, 1] -= f_min_y

        canvas = canvas[f_min_y:f_max_y, f_min_x:f_max_x]
        final_mask = final_mask[f_min_y:f_max_y, f_min_x:f_max_x]
        final_obj_mask = final_obj_mask[f_min_y:f_max_y, f_min_x:f_max_x]

        if canvas.shape[0] > canvas.shape[1] * 2:
            diff = canvas.shape[0] - canvas.shape[1] * 2
            diff = diff//4

            canvas = np.pad(canvas, ((0, 0), (diff, diff), (0, 0)), 'constant', constant_values = 1) 
            final_mask = np.pad(final_mask, ((0, 0), (diff, diff), (0, 0)), 'constant', constant_values = 1)
            final_obj_mask = np.pad(final_obj_mask, ((0, 0), (diff, diff), (0, 0)), 'constant', constant_values = 0)
            new_keypoints[:, 0] += diff

        elif canvas.shape[0] < canvas.shape[1] * 2:
            diff = canvas.shape[1] * 2 - canvas.shape[0]
            diff = diff//2
            
            canvas = np.pad(canvas, ((diff, diff), (0, 0), (0, 0)), 'constant', constant_values = 1)
            final_mask = np.pad(final_mask, ((diff, diff), (0, 0), (0, 0)), 'constant', constant_values = 1)
            final_obj_mask = np.pad(final_obj_mask, ((diff, diff), (0, 0), (0, 0)), 'constant', constant_values = 0)
            new_keypoints[:, 1] += diff
 
        new_keypoints[:, 1] *= (896/canvas.shape[0])
        new_keypoints[:, 0] *= (448/canvas.shape[1])
        new_keypoints = np.int32(new_keypoints)
        new_keypoints[:, 1] += 64
        new_keypoints[:, 0] += 32

        canvas = cv2.resize(canvas, (448, 896))
        canvas = np.pad(canvas, ((64, 64), (32, 32), (0, 0)), 'constant', constant_values = 1)
        
        final_mask = cv2.resize(final_mask, (448, 896))
        final_mask = np.pad(final_mask, ((64, 64), (32, 32), (0, 0)), 'constant', constant_values = 1)
 
        final_obj_mask = cv2.resize(final_obj_mask, (448, 896))
        final_obj_mask = np.pad(final_obj_mask, ((64, 64), (32, 32), (0, 0)), 'constant', constant_values = 0)

        body_pose_canvas = copy.deepcopy(canvas)
        body_pose_canvas = util.draw_bodypose(body_pose_canvas, new_keypoints, new_subset)

        plt.imshow(body_pose_canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        #plt.show() 

        test_keypoints = []

        for i in new_subset[0][:-2]:
            if i == -1:
                test_keypoints.append(None)
            else:
                test_keypoints.append(controlnet_aux.open_pose.body.Keypoint(new_keypoints[int(i)][0], new_keypoints[int(i)][1]))

        blank2 = np.zeros(canvas.shape)
        blank2 = controlnet_aux.open_pose.util.draw_bodypose(blank2, test_keypoints)

        plt.imshow(blank2[:, :, [2, 1, 0]]*255)
        plt.axis('off')
        #plt.show()

        cv2.imwrite(f'{args.output_folder}/{image_name}.png', np.uint8(canvas * 255))
        cv2.imwrite(f'{args.output_folder}/{image_name}_keypoints.jpg', np.uint8(blank2[:, :, ::-1]))
        cv2.imwrite(f'{args.output_folder}/{image_name}_mask.png', np.uint8(final_mask * 255))
        cv2.imwrite(f'{args.output_folder}/{image_name}_obj_mask.png', np.uint8(final_obj_mask * 255))
        if os.path.isdir(f'{args.output_folder}/lama/'):
            shutil.rmtree(f'{args.output_folder}/lama/')
        os.mkdir(f'{args.output_folder}/lama/')
        cv2.imwrite(f'{args.output_folder}/lama/image001.png', np.uint8(canvas * 255))
        cv2.imwrite(f'{args.output_folder}/lama/image001_mask.png', np.uint8(final_obj_mask * 255))
        
        print(f' -- Finished processing \'{image_name}\'. --')

        device = torch.device("cpu")
        model_path = "LaMa_models/big-lama/"

        train_config_path = model_path + "config.yaml"
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = ".png"
        checkpoint_path = model_path + "models/best.ckpt"

        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to('cpu')

        dataset = make_default_val_dataset(indir = f'{args.output_folder}/lama/', kind = "default", img_suffix = ".png", pad_out_to_modulo = 8)
        for img_i in trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = f'{args.output_folder}/{image_name}_lama.png'
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            cur_res = refine_predict(batch, model, gpu_ids="0,", modulo=8, n_iters=10, lr=0.002, min_side=256, max_scales=3, px_budget = 1800000)
            cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
        
        if diff < 1:

            generator = torch.Generator(device="cpu").manual_seed(1)

            init_image = Image.open(f"{args.output_folder}/{image_name}.png")
            mask_image = Image.open(f"{args.output_folder}/{image_name}_mask.png")
            control_image = Image.open(f"{args.output_folder}/{image_name}_keypoints.png")

            controlnet_inpaint = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint")

            controlnet_openpose = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose")

            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet_openpose, safety_checker = None).to('cpu')

            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            # generate image
            image = pipe(
                "a delivery man in a red shirt wearing a red cap",
                num_inference_steps=20,
                generator=generator,
                eta=1.0,
                image=init_image,
                mask_image=mask_image,
                control_image=control_image,
            ).images[0]

            image.save(f'{args.output_folder}/{image_name}_SD.png')

            run_PTI(f'{args.output_folder}/{image_name}_SD.png', f'{args.output_folder}/{image_name}_mask.png') 

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    t1 = time.time()
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'StyleGAN-Human data process'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)
    parser.add_argument('--image-folder', type=str, dest='image_folder')
    parser.add_argument('--output-folder', dest='output_folder', default='results', type=str)
    # parser.add_argument('--cfg', dest='cfg for segmentation', default='PP_HumanSeg/export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml', type=str)

    print('parsing arguments')
    cmd_args = parser.parse_args()
    run(cmd_args)

    print('total time elapsed: ', str(time.time() - t1))
