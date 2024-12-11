import pyrender
import pyrender.camera as cam
import torch
import numpy as np
import trimesh
import typing

__all__ = [
    'HDRenderer'
]

COLORS = {
    'N': [1.0, 1.0, 0.9],
    'GT': [146 / 255.0, 189 / 255.0, 163 / 255.0]
}

class WeakPerspectiveCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 scale,
                 translation,
                 znear=cam.DEFAULT_Z_NEAR,
                 zfar=cam.DEFAULT_Z_FAR,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        P[0, 3] = self.translation[0] * self.scale
        P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1

        return P

class AbstractRenderer(object):
    def __init__(self, faces=None, img_size=224, use_raymond_lighting=True):
        super(AbstractRenderer, self).__init__()

        self.img_size = img_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size,
            viewport_height=img_size,
            point_size=1.0
        )
        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.0, 0.0, 0.0))
        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

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
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False, deg=0):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces, process=False)

        curr_vertices = vertices.copy()
        mesh = self.mesh_constructor(
            curr_vertices, faces, process=False)
        if deg != 0:
            rot = self.transf(
                np.radians(deg), [0, 1, 0],
                point=np.mean(curr_vertices, axis=0))
            mesh.apply_transform(rot)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces, body_color=(1.0, 1.0, 1.0, 1.0),
                    deg=0):
        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=body_color, deg=deg)
        self.scene.add(body_mesh, name='body_mesh')

class OverlayRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224, tex_size=1):
        super(OverlayRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, scale, translation):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = WeakPerspectiveCamera(scale, translation,
                                   znear=1e-5,
                                   zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_scale, camera_translation, bg_imgs=None,
                 deg=0,
                 return_with_alpha=False,
                 body_color=None,
                 **kwargs):

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_scale):
            camera_scale = camera_scale.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W

            self.update_camera(camera_scale[bidx], camera_translation[bidx])
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)

class HDRenderer(OverlayRenderer):
    def __init__(self, **kwargs):
        super(HDRenderer, self).__init__(**kwargs)

    def update_camera(self, focal_length, translation, center):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=center[0],
            cy=center[1],
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = translation.copy()
        camera_pose[0, 3] *= (-1)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self,
        vertices: torch.Tensor,
        faces: typing.Union[torch.Tensor, np.ndarray],
        focal_length: typing.Union[torch.Tensor, np.ndarray],
        camera_translation: typing.Union[torch.Tensor, np.ndarray],
        camera_center: typing.Union[torch.Tensor, np.ndarray],
        bg_imgs: np.ndarray,
        render_bg: bool = True,
        deg: float = 0,
        return_with_alpha: bool = False,
        body_color: typing.List[float] = None,
        **kwargs
    ):
        '''
            Parameters
            ----------
            vertices: BxVx3, torch.Tensor
                The torch Tensor that contains the current vertices to be drawn
            faces: Fx3, np.ndarray
                The faces of the meshes to be drawn. Right now only support a
                batch of meshes with the same topology
            focal_length: B, torch.Tensor
                The focal length used by the perspective camera
            camera_translation: Bx3, torch.Tensor
                The translation of the camera estimated by the network
            camera_center: Bx2, torch.Tensor
                The center of the camera in pixels
            bg_imgs: np.ndarray
                Optional background images used for overlays
            render_bg: bool, optional
                Render on top of the background image
            deg: float, optional
                Degrees to rotate the mesh around itself. Used to render the
                same mesh from multiple viewpoints. Defaults to 0 degrees
            return_with_alpha: bool, optional
                Whether to return the rendered image with an alpha channel.
                Default value is False.
            body_color: list, optional
                The color used to render the image.
        '''
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        if torch.is_tensor(focal_length):
            focal_length = focal_length.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        if torch.is_tensor(camera_center):
            camera_center = camera_center.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            _, H, W = bg_imgs[bidx].shape
            # Update the renderer's viewport
            self.renderer.viewport_height = H
            self.renderer.viewport_width = W

            self.update_camera(
                focal_length=focal_length[bidx],
                translation=camera_translation[bidx],
                center=camera_center[bidx],
            )
            self.update_mesh(
                vertices[bidx], faces, body_color=body_color, deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if render_bg:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    # output_img = (color * valid_mask +
                    #               (1 - valid_mask) * curr_bg_img)
                                  
                    output_img = np.where(valid_mask, 
                        color * 0.65 + 0.35 * curr_bg_img,
                        curr_bg_img
                    )
                    output_imgs.append(np.clip(output_img, 0, 1))                    
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
            else:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
        return np.stack(output_imgs, axis=0)