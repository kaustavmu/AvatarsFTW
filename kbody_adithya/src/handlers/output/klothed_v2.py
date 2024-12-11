from collections.abc import Callable

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
import toolz

__all__ = ['KlothedBodyV2']

class KlothedBodyV2(Callable):
    def __init__(self,
        focal_length:               typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:            typing.Optional[typing.Union[float, typing.Tuple[float, float]]]=None,
        scale:                      float=1.0,
        blend:                      float=0.65,
        pad_scale:                  float=2.5,
        shoulder_scale:             float=0.75,
        landmark_perc_threshold:    float=0.1,
        joints2d:                   str='joints2d',
        joints3d:                   str='smplx_joints',
        j3d_head_index:             int=0,
        mirror:                     bool=False,
    ) -> None:
        super().__init__()
        self.focal_length = (float(focal_length), float(focal_length)) \
            if isinstance(focal_length, float) or isinstance(focal_length, int) else focal_length
        self.principal_point = (float(principal_point), float(principal_point)) \
            if isinstance(principal_point, float) or isinstance(principal_point, int) else principal_point
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=(0.7, 0.3, 0.1, 1.0)
        )
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        for light in self._create_raymond_lights():
            self.scene.add_node(light)
        self.scale = scale
        self.blend = blend
        self.pad_scale = pad_scale
        self.shoulder_scale = shoulder_scale
        self.landmark_perc_threshold = landmark_perc_threshold
        self.j3d_head_index = j3d_head_index
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.mirror = mirror
        self.renderer = None

    def save_body_legacy(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
    ):
        outs = []
        for i in range(len(jsons)):
            with open(jsons[i]['body']['body_legacy_t'], 'wb') as output_file:
                pickle.dump({                    
                    'camera_rotation': np.eye(3, dtype=np.float32)[np.newaxis, ...],
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
                }, output_file)
            outs.append({ 'body_legacy_t': 'Success' })
        return outs

    def save_body(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
    ):
        outs = []
        for i in range(len(jsons)):
            with open(jsons[i]['body']['body_t'], 'wb') as output_file:
                pickle.dump({                    
                    'camera_rotation': np.eye(3, dtype=np.float32),
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
                }, output_file)
            outs.append({ 'body_t': 'Success' })
        return outs

    def save_body_padded(self, 
        data:   typing.Mapping[str, torch.Tensor],
        jsons:   typing.Mapping[str, typing.Any],
        transl: np.array,
    ):
        outs = []
        for i in range(len(jsons)):
            with open(jsons[i]['body']['body_legacy_t'], 'wb') as output_file:
                pickle.dump({                    
                    'camera_rotation': np.eye(3, dtype=np.float32)[np.newaxis, ...],
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
                }, output_file)
            outs.append({ 'body_legacy_t': 'Success' })
        return outs

    def needs_adjustment(self, 
        color:                      torch.Tensor,
        joints2d:                   torch.Tensor, # openpose format
        joints3d:                   torch.Tensor, # openpose format ?
        translation:                np.array,
        top_head_y:                 float,
        focal_length:               float=5000.0,
        pad_scale:                  float=2.5,
        shoulder_scale:             float=0.75,
        landmark_perc_threshold:    float=0.1,
        j3d_head_index:             int=0,        
    ):
        j2d = joints2d.detach().cpu().numpy()
        img = color # .detach().cpu().numpy()
        j3d = joints3d.detach().cpu().numpy()
        j3d += translation
        head_root = j2d[0:1, :]
        head_other = j2d[15:19, :]
        face = j2d[67:, :]
        head_outside = head_root[:, 1] < 0 or np.any(head_other[:, 1] < 0) or top_head_y < 0
        face_outside_perc = (face[:, 1] < 0).sum() / face.shape[0]
        needs_adjustment = head_outside or face_outside_perc > landmark_perc_threshold
        FILL = [0, 0, 0] # [1.0, 1.0, 1.0] # [0, 0, 0]
        if needs_adjustment:
            rshoulder = j2d[2:3, :]
            lshoulder = j2d[5:6, :]
            shoulder_vector = rshoulder - lshoulder
            perpendicular_vector = np.flip(shoulder_vector, axis=-1)
            shoulder_centroid = (rshoulder + lshoulder) * 0.5
            estimated_head_position = shoulder_centroid + perpendicular_vector * shoulder_scale
            vertical_padding = int(np.abs(estimated_head_position[:, 1]) * pad_scale)
            if vertical_padding < abs(top_head_y) and top_head_y < 0:
                vertical_padding = int(2.0 * abs(top_head_y))
            if vertical_padding <= 0:
                vertical_padding = 0
            bottom = 0 if not self.mirror else vertical_padding
            padded = cv2.copyMakeBorder(img.copy(), 
                vertical_padding, bottom, 0, 0, cv2.BORDER_CONSTANT, value=FILL
            )
            return (padded, vertical_padding, (0.5 if not self.mirror else 0.0) * vertical_padding / focal_length * j3d[j3d_head_index, 2])
        else:
            return (img, 0.0, 0.0)

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
        for i in range(b):
            has_no_metadata = 'metadata_t' not in jsons[i]['body'] or not os.path.exists(jsons[i]['body']['metadata_t'])
            metadata = { } if has_no_metadata else self.load_metadata(jsons[i]['body']['metadata_t'])
            input_img = background.flip(dims=[1]).detach().cpu().numpy()[i].squeeze().transpose(1, 2, 0)
            translation = data['params']['translation_t'][i].detach().cpu().numpy().squeeze()
            v = data['body']['vertices'][0].detach().cpu().numpy().squeeze()
            f = data['body']['faces'][0].detach().cpu().numpy().squeeze()
            h3d = v[581] + translation #head top index: 581
            h2d_y = h3d[1] / h3d[2] * self.focal_length[1] + (oh * 0.5)
            input_img, dv, dy = self.needs_adjustment(input_img,
                data[self.joints2d][i], data[self.joints3d][i], translation,                
                h2d_y, self.focal_length[1], self.pad_scale, self.shoulder_scale,
                self.landmark_perc_threshold, self.j3d_head_index
            )
            h, w, c = input_img.shape
            rotation = data['camera_rotation'][i].detach().cpu().numpy().squeeze() \
                if 'camera_rotation' in data else np.eye(3)                     
            tmesh = trimesh.Trimesh(v, f, process=False)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            tmesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(tmesh, material=self.material)
            node = self.scene.add(mesh, 'mesh')
            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            translation[0] *= -1.0
            translation[1] += dy
            translations.append(translation)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation
            # camera_pose = np.linalg.inv(camera_pose)
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
                color[:, :, :-1] * self.blend + (1.0 - self.blend) * input_img,
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
            if dv != 0.0:
                metadata['head_found'] = True
                metadata['mirrored'] = self.mirror
                metadata['dv'] = dv
                metadata['dy'] = dy
                cv2.imwrite(jsons[i]['body']['padded_t'], (input_img * 255.0).astype(np.uint8))
                # input_img = cv2.resize(input_img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                # cv2.imwrite(jsons[i]['body']['image'], (input_img * 255.0).astype(np.uint8))
                self.save_body_padded(data, jsons, np.stack(translations, axis=0))
                jsons_list = []
                for j in jsons:
                    legacy_filename = j['body']['body_legacy_t']
                    jsons_list.append({'body': toolz.assoc(
                        toolz.dissoc(j['body'], 'body_legacy_t'), 
                        'body_legacy_t',
                        legacy_filename.replace('.pkl', '_original.pkl')
                    )})
                # self.save_body_legacy(data, jsons_list)
                valid_mask[:dv, :, :] = True
                valid_mask[dv:, :, :] = False
                if self.mirror:
                    valid_mask[-dv:, :, :] = True
                padded_filename = jsons[i]['body']['padded_t']
                filename, ext = os.path.splitext(padded_filename)
                cv2.imwrite(filename + "_mask" + ext, valid_mask.astype(np.uint8) * 255)
                outs.append({ 'code': 200, 'message': 'Missing head detected' })
            elif 'dv' in metadata:
                input_img = input_img[:-metadata['dv'], :, :]
                cv2.imwrite(jsons[i]['body']['padded_t'], (input_img * 255.0).astype(np.uint8))                
                with open(jsons[i]['body']['openpose'], 'r') as keypoint_file:
                    data = json.load(keypoint_file)                
                offset = np.array([[0.0, metadata['dv'], 0.0]], dtype=np.float32)
                for person in data['people']:
                    person['pose_keypoints_2d'] = (
                        np.array(person['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    
                    person['hand_left_keypoints_2d'] = (
                        np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    person['hand_right_keypoints_2d'] = (
                        np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                    person['face_keypoints_2d'] = (
                        np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3]) - offset
                    ).reshape(-1).tolist()
                with open(os.path.basename(jsons[i]['body']['openpose']).replace('.json', '_t.json'), 'w') as f:
                    json.dump(data, f)
                outs.append({ 'code': 200, 'message': 'Padded head adjusted' })
            else:
                metadata['head_found'] = True
                self.save_body_legacy(data, jsons)
                outs.append({ 'code': 200, 'message': 'Success' })
            np.savez(jsons[i]['body']['metadata_t'], metadata=metadata)
        renderer.delete()
        return outs

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