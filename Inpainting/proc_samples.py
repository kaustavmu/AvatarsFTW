import os
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from openpose.src import util
from openpose.src.body import Body

body_estimation = Body('openpose/model/body_pose_model.pth')

image_files = os.listdir("SHHQ-1.0_samples/")

top_Y = []
shoulder_Y = []
hip_Y = []
knee_Y = []
feet_Y = []

L_shoulder_X = []
R_shoulder_X = []
L_knee_X = []
R_knee_X = []

l = len(image_files)

full_res = []

for image_file in tqdm(image_files):
    
    image = cv2.imread("SHHQ-1.0_samples/" + image_file)
    keypoints, subset = body_estimation(image)
    subset = np.int8(subset[0][:-2])

    min_X = min(keypoints[:, 0])
    max_X = max(keypoints[:, 0])
    min_Y = min(keypoints[:, 1])
    max_Y = max(keypoints[:, 1])
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

    full_res.append(res)

    if subset[14] != -1 and subset[15] != -1:
        Y = (keypoints[subset[14]][1] + keypoints[subset[15]][1])/2
        Y = (Y-min_Y)/h
        top_Y.append(Y) 
    if subset[10] != -1 and subset[13] != -1:
        Y = (keypoints[subset[10]][1] + keypoints[subset[13]][1])/2
        Y = (Y-min_Y)/h
        feet_Y.append(Y) 
    if subset[9] != -1 and subset[12] != -1:
        Y = (keypoints[subset[9]][1] + keypoints[subset[12]][1])/2
        Y = (Y-min_Y)/h
        knee_Y.append(Y)
    if subset[8] != -1 and subset[11] != -1:
        Y = (keypoints[subset[8]][1] + keypoints[subset[11]][1])/2
        Y = (Y-min_Y)/h
        hip_Y.append(Y)
    if subset[2] != -1 and subset[5] != -1:
        Y = (keypoints[subset[2]][1] + keypoints[subset[5]][1])/2
        Y = (Y-min_Y)/h
        shoulder_Y.append(Y)

    if subset[2] != -1:
        X = keypoints[subset[2]][0]
        X = (X-min_X)/w
        L_shoulder_X.append(X)
    if subset[5] != -1:
        X = keypoints[subset[5]][0]
        X = (X-min_X)/w
        R_shoulder_X.append(X)
    if subset[9] != -1:
        X = keypoints[subset[9]][0]
        X = (X-min_X)/w
        L_knee_X.append(X)
    if subset[12] != -1:
        X = keypoints[subset[12]][0]
        X = (X-min_X)/w
        R_knee_X.append(X)

full_res = np.array(full_res)
np.save("samples.npy", full_res)

print("top_Y", np.mean(top_Y))
print("shoulder_Y", np.mean(shoulder_Y))
print("hip_Y", np.mean(hip_Y))
print("knee_Y", np.mean(knee_Y))
print("feet_Y", np.mean(feet_Y))
print("L_shoulder_X", np.mean(L_shoulder_X))
print("R_shoulder_X", np.mean(R_shoulder_X))
print("L_knee_X", np.mean(L_knee_X))
print("R_knee_X", np.mean(R_knee_X))
