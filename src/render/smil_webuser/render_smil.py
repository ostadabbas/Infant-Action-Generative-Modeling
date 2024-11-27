'''
This script is adapted from the SMPL package, copyright 2015 Matthew Loper, Naureen Mahmood and
the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMIL is available here http://s.fhg.de/smil
For comments or questions, please email us at: smil@iosb.fraunhofer.de


Please Note:
============
This is a demo version of the script for driving the SMIL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the SMIL model and rendering it using OpenDR
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMIL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smil_webuser/hello_world directory. You can run
the hello world script now by typing the following:
>	python render_smil.py


'''
## Only needed for pose prior
class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)


import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smil_webuser.serialization import load_model
from cPickle import load

## Load SMIL model
m = load_model('../../smil_web.pkl')

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose[0] = np.pi

## Alternatively assign mean pose from pose prior
prior_filename = '../../smil_pose_prior.pkl'
prior = load(open(prior_filename, 'r'))
m.pose[3:] = prior.mean # first three pose parameters contain global rotation

## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, -.1, .5]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': .1, 'far': 2., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-100,-50,-200]),
    vc=np.ones_like(m)*.9,
    light_color=np.array([1., 1., 1.]))


## Show it using OpenCV
import cv2
cv2.imshow('render_SMIL', rn.r)
print ('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()
