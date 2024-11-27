import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from .renderer_samples import get_renderer_samples

import cv2
from opendr.simple import *
from opendr.renderer import ColoredRenderer
from opendr.renderer import TexturedRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from .smil_webuser import *
from .smil_webuser.serialization import load_model

def get_rotation(theta=np.pi/3):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(meshes, key, action, savepath, savefolder):
    print(savepath)

    w, h = (640, 480)

    writer = imageio.get_writer(savepath, fps=30)
    # center the first frame
    meshes = meshes - meshes[0].mean(axis=0)
    # matrix = get_rotation(theta=np.pi/4)
    # meshes = meshes[45:]
    # meshes = np.einsum("ij,lki->lkj", matrix, meshes)

    bg_file = '/home/faye/Documents/smil/bg_img/infant_bg/bg3.jpg'
    bg = cv2.imread(bg_file)
    #bg_im = bg.astype(np.float64)/255
    bg_h = bg.shape[0]
    bg_w = bg.shape[1]
    #print(bg_w)
    x = 0
    y = 0
    bg_im = bg[y:y+h, x:x+w].astype(np.float64)/255

    txt_file = '/home/faye/Documents/smil/textures/infant_txt/txt2.png'
    txt = cv2.imread(txt_file)
    txt = cv2.cvtColor(txt, cv2.COLOR_BGR2RGB)
    txt_im = txt.astype(np.float64)/255 

    imgs = []

    idx_frame = 0
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        idx_frame += 1
        m, kin_table = load_model('/home/faye/Documents/smil/smil_web.pkl')
        tmpl = load_mesh('/home/faye/Documents/smil/template.obj')
        #m.pose = mesh.flatten()

        rt = np.zeros(3)
        t = np.array([0, 0.1, 0.8])

        cam = ProjectPoints(v=m.r, rt=rt, t = t, f=np.array([w,w])/2.5, c=np.array([w,h])/2, k=np.zeros(5))
  
        rn = TexturedRenderer(v=m.r, f=m.f, vc=np.ones_like(m.r), vt=tmpl.vt, ft=tmpl.ft,
                                     texture_image = txt_im, background_image = bg_im,
                                     camera = cam,
                                     frustum = {'near': 0.15, 'far': 1.5, 'width': w, 'height': h},
                                     overdraw=False)
    
        data = 255 * rn.r
        imgs.append(data)

        file_name = os.path.join(savefolder, str(idx_frame) + '.jpg')
        cv2.imwrite(file_name, data)
        #show(img)

    imgs = np.array(imgs)
    '''
    masks = ~(imgs/255. > 0.96).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    for cimg in imgs[:, y1-20:y2+20, x1-20:x2+20]:
        writer.append_data(cimg)
    '''
    for img in imgs:
        writer.append_data(img)  
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    opt = parser.parse_args()
    filename = opt.filename
    savefolder = os.path.splitext(filename)[0]
    os.makedirs(savefolder, exist_ok=True)
    savefolder = os.path.join(savefolder, 'examples')
    os.makedirs(savefolder, exist_ok=True)

    generation = np.load(filename, allow_pickle=True).item()

    nframes, num_joints, _ = generation['pose'][0].shape
    nspa = 50
    nats = 5
    motions_array = np.array(generation['pose'])
    labels_array = np.array(generation['y'])

    # Initialize a new array with the desired shape: 50x5x24x3x60
    gener = np.zeros((nspa, nats, num_joints, 3, nframes))

    # Populate the new array
    for k in range(nats):  # For each class
        class_samples = motions_array[labels_array == k]  # Get samples for the current class
        class_samples = class_samples.transpose(0, 2, 3, 1)  # Reshape from 50x60x24x3 to 50x24x3x60
        gener[:, k, :, :, :] = class_samples

    action = 0
    idx = 0
    key = 'gen'
    poses = gener[idx][action] 
    print(poses.shape)
    meshes = poses.transpose(2, 0, 1)
    path = os.path.join(savefolder, "action{}_{}_{}.mp4".format(action, idx, key))
    render_video(meshes, key, action, path, savefolder)

if __name__ == "__main__":
    main()
