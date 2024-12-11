from collections.abc import Callable

import toolz
import json
import os
import typing
import torch
import pyrender
import numpy as np
import trimesh
import cv2
from PIL import Image
import pickle
import math
import logging

__all__ = ['KlothedBodyV4']

log = logging.getLogger(__name__)

class KlothedBodyV4(Callable):

    _GENDERS_ = ['male', 'female']
    __VERSION__ = 4.0

    def __init__(self,
        focal_length:               typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:            typing.Optional[typing.Union[float, typing.Tuple[float, float]]]=None,
        scale:                      float=1.0,
        blend:                      float=0.65,
        joints3d:                   str='smplx_joints',
        j3d_head_index:             int=0,
        has_decomposed_betas:       bool=False,
        gender:                     str='neutral',
        height_regressors:          str='',        
    ) -> None:
        super().__init__()
        self.focal_length = (float(focal_length), float(focal_length)) \
            if isinstance(focal_length, float) or isinstance(focal_length, int) else focal_length
        self.principal_point = (float(principal_point), float(principal_point)) \
            if isinstance(principal_point, float) or isinstance(principal_point, int) else principal_point
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=(0.7, 0.25, 0.5, 1.0)
        )
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        for light in self._create_raymond_lights():
            self.scene.add_node(light)
        self.scale = scale
        self.blend = blend
        self.j3d_head_index = j3d_head_index
        self.joints3d = joints3d
        self.has_decomposed_betas = has_decomposed_betas
        self.gender = gender
        self.renderer = None
        self.height_regressors = {}
        if height_regressors and os.path.exists(height_regressors):
            male = os.path.join(height_regressors, 'male.pkl')
            if os.path.exists(male):
                with open(male, 'rb') as f:
                    self.height_regressors['male'] = pickle.load(f)
            else:
                log.warning(f"Could not find {male} in {height_regressors}, will skip height calculation for males.")
            female = os.path.join(height_regressors, 'female.pkl')
            if os.path.exists(female):
                with open(female, 'rb') as f:
                    self.height_regressors['female'] = pickle.load(f)
            else:
                log.warning(f"Could not find {female} in {height_regressors}, will skip height calculation for females.")
            neutral = os.path.join(height_regressors, 'neutral.pkl')
            if os.path.exists(neutral):
                with open(neutral, 'rb') as f:
                    self.height_regressors['neutral'] = pickle.load(f)
            else:
                log.warning(f"Could not find {neutral} in {height_regressors}, will skip height calculation for neutral.")

    def calculate_ipd_based_height(self,
        data:       typing.Mapping[str, torch.Tensor],
        gender:     str='neutral',
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            the average adult IPD for women is 6.17 cm with a standard deviation of 0.36 cm, 
            and 6.40 cm for men with a standard deviation of 0.34 cm
            Because our 3D models do not have pupils, 
            we measure the center of the eye as the midway point between
            the left and right corners of the eye. 
            We observe empirically that this definition of IPD is slightly larger than the clinical
            definition, and therefore scale all measured IPDs by 0.975.
        '''
        ipd = data['ipd'].detach().cpu() * 0.975 * 100.0 # corrected cm
        stats = {
            'neutral': { 'avg': (6.4 + 6.17) * 0.5, 'std': (0.34 + 0.36) * 0.5, },
            'male': { 'avg': 6.4, 'std': 0.34, },
            'female': { 'avg': 6.17, 'std': 0.36, },
        }
        # men_avg, men_std = 6.4, 0.34
        # women_avg, women_std = 6.17, 0.36
        mesh_height = data['height'].detach().cpu()
        height = data['height'].detach().cpu() * (stats[gender]['avg'] / ipd) # in meters
        if gender in self.height_regressors:
            data = np.array([
                float(mesh_height), 
                float(ipd * 0.01), 
                float(ipd * 0.01 / mesh_height),
                float(ipd * 0.01 / stats[gender]['avg'])
            ])
            actual_height = float(self.height_regressors[gender].predict(data[np.newaxis, :]))
        else:
            actual_height = height
        log.info(f"Estimated Mesh IPD = {float(ipd):.4f}cm, Mesh Scaled Height = {float(height):.4f}m, Actual Height = {float(actual_height):.4f}")
        return (
            torch.scalar_tensor(float(ipd)), 
            torch.scalar_tensor(float(height)),
            torch.scalar_tensor(float(mesh_height)),
            torch.scalar_tensor(float(actual_height)),
        )

    def save_body_legacy(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
    ):
        outs = []
        for i in range(len(jsons)):
            gender_id = toolz.get('gender', data, None)
            gender = 'neutral' if gender_id is None else toolz.get(int(gender_id[i]), KlothedBodyV4._GENDERS_, 'neutral')
            ipd, estimated_height, mesh_height, actual_height = self.calculate_ipd_based_height(data, gender)
            with open(jsons[i]['body']['body_legacy_t'], 'wb') as output_file:
                pickle.dump({                    
                    # 'camera_rotation': np.eye(3, dtype=np.float32)[np.newaxis, ...],
                    'camera_rotation': data['camera_rotation'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'leye_pose': np.zeros((1, 3), dtype=np.float32),
                    'reye_pose': np.zeros((1, 3), dtype=np.float32),
                    'global_orient': data['params']['global_orient_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'camera_translation': data['params']['translation_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'betas': data['params']['betas_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'jaw_pose': data['params']['jaw_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'right_hand_pose': data['params']['lhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'left_hand_pose': data['params']['rhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'expression': data['params']['expression_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'body_pose': data['params']['pose_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'ipd': ipd[np.newaxis, ...].detach().cpu().numpy(),
                    'estimated_height': estimated_height[np.newaxis, ...].detach().cpu().numpy(),
                    'mesh_height': mesh_height[np.newaxis, ...].detach().cpu().numpy(),
                    'actual_height': actual_height[np.newaxis, ...].detach().cpu().numpy(),
                    'version': KlothedBodyV4.__VERSION__,
                }, output_file)
            outs.append({ 'body_legacy_t': 'Success' })
        return outs

    def save_body(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
    ):
        outs = []
        for i in range(len(jsons)):
            gender_id = toolz.get('gender', data, None)
            gender = 'neutral' if gender_id is None else toolz.get(int(gender_id[i]), KlothedBodyV4._GENDERS_, 'neutral')
            ipd, estimated_height, mesh_height, actual_height = self.calculate_ipd_based_height(data, gender)
            with open(jsons[i]['body']['body_t'], 'wb') as output_file:
                pickle.dump({                    
                    # 'camera_rotation': np.eye(3, dtype=np.float32),
                    'camera_rotation': data['camera_rotation'][i].detach().cpu().numpy(),
                    'leye_pose': np.zeros(3, dtype=np.float32),
                    'reye_pose': np.zeros(3, dtype=np.float32),
                    'global_orient': data['params']['global_orient_t'][i].detach().cpu().numpy(),
                    'camera_translation': data['params']['translation_t'][i].detach().cpu().numpy(),
                    'latent_pose': data['params']['pose_t'][i].detach().cpu().numpy(),
                    'betas': data['params']['betas_t'][i].detach().cpu().numpy(),
                    'jaw_pose': data['params']['jaw_t'][i].detach().cpu().numpy(),
                    'right_hand_pose': data['params']['lhand_t'][i].detach().cpu().numpy(),
                    'left_hand_pose': data['params']['rhand_t'][i].detach().cpu().numpy(),
                    'expression': data['params']['expression_t'][i].detach().cpu().numpy(),
                    'body_pose': data['body']['pose'][i].detach().cpu().numpy(),
                    'ipd': ipd[np.newaxis, ...].detach().cpu().numpy(),
                    'estimated_height': estimated_height[np.newaxis, ...].detach().cpu().numpy(),
                    'mesh_height': mesh_height[np.newaxis, ...].detach().cpu().numpy(),
                    'actual_height': actual_height[np.newaxis, ...].detach().cpu().numpy(),
                    'version': KlothedBodyV4.__VERSION__,
                }, output_file)
            if 'betas_t' in jsons[i]['body']:
                np.savetxt(jsons[i]['body']['betas_t'], data['params']['betas_t'][i].detach().cpu().numpy().squeeze()[:20])
            outs.append({ 'body_t': 'Success' })            
        return outs

    def save_body_padded(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
        transl: np.array,
    ):
        outs = []
        for i in range(len(jsons)):
            gender_id = toolz.get('gender', data, None)
            gender = 'neutral' if gender_id is None else toolz.get(int(gender_id[i]), KlothedBodyV4._GENDERS_, 'neutral')
            ipd, estimated_height, mesh_height, actual_height = self.calculate_ipd_based_height(data, gender)
            with open(jsons[i]['body']['body_legacy_t'], 'wb') as output_file:
                pickle.dump({                    
                    # 'camera_rotation': np.eye(3, dtype=np.float32)[np.newaxis, ...],
                    'camera_rotation': data['camera_rotation'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'leye_pose': np.zeros((1, 3), dtype=np.float32),
                    'reye_pose': np.zeros((1, 3), dtype=np.float32),
                    'global_orient': data['params']['global_orient_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'camera_translation': transl[i],
                    'betas': data['params']['betas_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'jaw_pose': data['params']['jaw_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'right_hand_pose': data['params']['lhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'left_hand_pose': data['params']['rhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'expression': data['params']['expression_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'body_pose': data['params']['pose_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'ipd': ipd[np.newaxis, ...].detach().cpu().numpy(),
                    'estimated_height': estimated_height[np.newaxis, ...].detach().cpu().numpy(),
                    'mesh_height': mesh_height[np.newaxis, ...].detach().cpu().numpy(),
                    'actual_height': actual_height[np.newaxis, ...].detach().cpu().numpy(),
                    'version': KlothedBodyV4.__VERSION__,
                }, output_file)
            outs.append({ 'body_legacy_t': 'Success' })
        return outs            

    def load_metadata(self,
        filename: str    
    ) -> dict:
        return np.load(filename, allow_pickle=True)['metadata'].item()

    def save_metadata(self,
        filename: str,
        data: dict,        
    ) -> dict:
        return np.savez(filename, metadata=data)

    def __call__(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        outs = []            
        background = data['color']
        b, c, oh, ow = background.shape
        translations = []
        if self.has_decomposed_betas and 'betas_t' in data and 'params' in data:
            data['params']['betas_t'] = data['betas_t']
        for i in range(b):
            has_no_metadata = 'metadata_t' not in jsons[i]['body'] or not os.path.exists(jsons[i]['body']['metadata_t'])
            metadata = { } if has_no_metadata else self.load_metadata(jsons[i]['body']['metadata_t'])
            input_img = background.flip(dims=[1]).detach().cpu().numpy()[i].squeeze().transpose(1, 2, 0)
            translation = data['params']['translation_t'][i].detach().cpu().numpy().squeeze()
            v = data['body']['vertices'][0].detach().cpu().numpy().squeeze()
            f = data['body']['faces'][0].detach().cpu().numpy().squeeze()            
            rotation = data['camera_rotation'][i].detach().cpu().numpy().squeeze() \
                if 'camera_rotation' in data else np.eye(3)
            v = np.einsum('ij,pj->pi', rotation, v)
            tmesh = trimesh.Trimesh(v, f, process=False)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            tmesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(tmesh, material=self.material)
            node = self.scene.add(mesh, 'mesh')
            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            '''
                'score': array(0., dtype=float32)
                'face': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
                'location_test': True
                'size_test': True
                'missing_head': True
                'top_padding': 175
                'bottom_padding': 175
                len(): 7
            '''
            if metadata.get('missing_head', False):
                bottom_pad = metadata['bottom_padding']
                j3d_head = data[self.joints3d][i][self.j3d_head_index].detach().cpu().numpy()
                j3d_head += translation
                j3d_head_z = j3d_head[2]
                dy = 0.5 * bottom_pad / self.focal_length[1] * j3d_head_z
                # dy *= oh / (oh - bottom_pad)
            else:
                bottom_pad = 0
                dy = 0.0

            padded_translation = translation.copy()
            padded_translation[1] += dy
            translations.append(padded_translation)
            camera_pose = np.eye(4)
            # rotation = np.linalg.inv(rotation)
            # camera_pose[:3, :3] = rotation
            render_translation = translation.copy()
            render_translation[0] *= -1.0
            render_translation[1] += dy
            camera_pose[:3, 3] = render_translation
            
            if dy != 0.0:
                oh, ow, _ = input_img.shape
                input_img = input_img[:-bottom_pad, :, :]
                nh = input_img.shape[0]
                # input_img = cv2.resize(input_img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(jsons[i]['body']['padded_t'], (input_img * 255.0).astype(np.uint8))
                matte_img = cv2.imread(jsons[i]['body']['matte'], cv2.IMREAD_ANYDEPTH)
                matte_img = matte_img[:-bottom_pad, ...]
                matte_img = matte_img.astype(np.float32)
                # matte_img = cv2.resize(matte_img, (ow, oh), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(jsons[i]['body']['matte_t'], matte_img)
                self.save_body_padded(data, jsons, np.stack(translations, axis=0))
                with open(jsons[i]['body']['keypoints'], 'r') as keypoint_file:
                    openpose_data = json.load(keypoint_file)
                # new_h = 
                offset = np.array([[0.0, bottom_pad, 0.0]], dtype=np.float32)
                offset = np.zeros_like(offset)
                scale = np.array([[1.0, oh / nh, 1.0]], dtype=np.float32)
                scale = np.ones_like(scale)
                for person in openpose_data['people']:
                    person['pose_keypoints_2d'] = (
                        scale * np.array(person['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    person['hand_left_keypoints_2d'] = (
                        scale * np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    person['hand_right_keypoints_2d'] = (
                        scale * np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    person['face_keypoints_2d'] = (
                        scale * np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                input_img = self.draw_openpose(input_img.copy(), openpose_data)
                # with open(os.path.basename(jsons[i]['body']['keypoints']).replace('.json', '_t.json'), 'w') as f:
                #     json.dump(openpose_data, f)
                outs.append({ 'code': 200, 'message': 'Unpadded' })
            else:
                self.save_body_legacy(data, jsons)
                outs.append({ 'code': 200, 'message': 'Success' })
            h, w, c = input_img.shape
            if self.principal_point is None:
                cx = w // 2
                cy = h // 2
            else:
                px, py = self.principal_point
                cx = px if px > 1.0 else px * w
                cy = py if py > 1.0 else py * h                    
            camera = pyrender.camera.IntrinsicsCamera(
                fx=self.focal_length[0], cx=cx,
                fy=self.focal_length[1], cy=cy,
            )
            cam = self.scene.add(camera, pose=camera_pose)
            renderer = pyrender.OffscreenRenderer(
                viewport_width=w, viewport_height=h, point_size=1.0
            )
            color, _ = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]            
            # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
            output_img = np.where(valid_mask, 
                color[:, :, :3] * self.blend + (1.0 - self.blend) * input_img,
                input_img
            )
            if self.scale != 1.0:
                output_img = np.array(
                    Image.fromarray(
                        output_img
                    ).resize(
                        (int(w * self.scale), int(h * self.scale)), Image.ANTIALIAS
                    )
                )
            self.scene.remove_node(node)
            self.scene.remove_node(cam)
            cv2.imwrite(jsons[i]['body']['overlay_t'], (output_img * 255.0).astype(np.uint8))
            self.save_body(data, jsons)
        renderer.delete()
        return outs

    def draw_openpose(self, img: np.ndarray, openpose_data: dict) -> None:
        poses = [np.array(openpose_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)]
        limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                            [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
        njoint = 25

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
                  [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
        for i in range(njoint):
            for n in range(len(poses)):
                pose = poses[n][i]
                if pose[2] <= 0:
                    continue
                x, y = pose[:2]
                cv2.circle(img, (int(x), int(y)), 4, colors[i], thickness=-1)
        stickwidth = 4
        for pose in poses:
            for limb,color in zip(limbSeq, colors):
                p1 = pose[limb[0]]
                p2 = pose[limb[1]]
                if p1[2] <=0 or p2[2] <= 0:
                    continue
                cur_canvas = img.copy()
                X = [p1[1],p2[1]]
                Y = [p1[0],p2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, color)
                img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
   
        return img

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        nodes = []
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes