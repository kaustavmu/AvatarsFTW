from collections.abc import Callable

import typing
import torch
import pyrender
import numpy as np
import trimesh
import cv2
from PIL import Image

__all__ = ['Renderer']

class Renderer(Callable):
    def __init__(self,
        focal_length:       typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:    typing.Optional[typing.Union[float, typing.Tuple[float, float]]]=None,
        scale:              float=1.0,
        blend:              float=0.65,
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
        self.renderer = None
    
    # def _get_renderer(self, width: int, height: int) -> pyrender.OffscreenRenderer:
    #     if self.renderer is None or self.renderer.viewport_width != width\
    #         or self.renderer.viewport_height != height:
    #             self.renderer = pyrender.OffscreenRenderer(
    #                 viewport_width=width, viewport_height=height, point_size=1.0
    #             )
    #     return self.renderer

    def __call__(self, 
        data:   typing.Mapping[str, torch.Tensor],
        json:   typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        outs = []            
        background = data['color']
        b, c, h, w = background.shape
        # renderer = self._get_renderer(width=w, height=h)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=w, viewport_height=h, point_size=1.0
        )
        for i in range(b):
            rotation = data['camera_rotation'][i].detach().cpu().numpy().squeeze() \
                if 'camera_rotation' in data else np.eye(3)
            translation = data['params']['translation_t'][i].detach().cpu().numpy().squeeze()                
            tmesh = trimesh.Trimesh(
                data['body']['vertices'].detach().cpu().numpy().squeeze(),
                data['body']['faces'].detach().cpu().numpy().squeeze(),
                process=False
            )
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            tmesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(tmesh, material=self.material)
            node = self.scene.add(mesh, 'mesh')
            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            translation[0] *= -1.0
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
            color, _ = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = background.flip(dims=[1]).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
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
            cv2.imwrite(json[i]['body']['overlay_t'], (output_img * 255.0).astype(np.uint8))
            outs.append({ 'overlay_t': 'Success' })
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