"""
This script is borrowed from https://github.com/mkocabas/VIBE
 Adhere to their licence to use this script
 It has been modified
"""

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from pyrender import MetallicRoughnessMaterial, Texture
import os
from PIL import Image

os.environ['PYOPENGL_PLATFORM'] = 'egl'
SMPL_MODEL_DIR = "models/smpl/"


def get_smpl_faces():
    return np.load(os.path.join(SMPL_MODEL_DIR, "smplfaces.npy"))


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
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
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class RendererSamples:
    def __init__(self, background=None, resolution=(224, 224), bg_color=[0, 0, 0, 0.5], orig_img=False, wireframe=False):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose.copy())

        """ok
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        """

        # light_pose[:3, 3] = [0, -2, 2]
        # [droite, hauteur, profondeur camera]
        """
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        """

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            print("dddddddddddddddddddddddd Mesh has UV coordinates.")
        else:
            print("dddddddddddddddddddddddddddddd Mesh does not have UV coordinates.")
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        
        texture_path = '/home/faye/Documents/smil/textures/infant_txt/txt2.png'
        texture_image = Image.open(texture_path).convert('RGB')
        texture_data = np.array(texture_image)
        texture = Texture(source=texture_data, source_channels='RGB')
        material = MetallicRoughnessMaterial(baseColorTexture=texture)

        '''
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        '''
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA
        print(img.shape)

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)

        alpha_mask = rgb[:,:,3] == 127
        fg_mask = rgb[:,:,3] != 127
        expanded_mask = np.repeat(alpha_mask[:,:,np.newaxis], 3, axis = 2)
        expanded_fg = np.repeat(fg_mask[:,:,np.newaxis], 3, axis = 2)

        output_img = rgb[:, :, :-1] * expanded_fg + img * expanded_mask

        '''
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis] 
        output_img = rgb[:, :, :-1] * valid_mask + (1 -  valid_mask) * img
        '''
        image = output_img.astype(np.uint8)
        #print(image.shape)
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def get_renderer_samples(bg, width, height):
    renderer = RendererSamples(bg, resolution=(width, height),
                        bg_color=[1, 1, 1, 0.5],
                        orig_img=False,
                        wireframe=False)
    return renderer