'''
Copyright 2020
Neng Qian, Jiayi Wang, Franziska Mueller, Florian Bernard,
Vladislav Golyanik, Christian Theobalt, and the Max Planck Institute.
All rights reserved.

This software is provided for research purposes only.
By using this software you agree to the terms of the HTML Model license.

More information about the HTML is available at https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/

Acknowledgements:
The code file is based on the release code of ICCV HAND2019 challenge with adaptations.
Check https://sites.google.com/view/hands2019/challenge
Therefore, we would like to kindly thank Dr. Anil Armagan.


Please Note:
============
This is a demo version of the script for driving the HTML, hand texture model with python.
We would be happy to receive comments, help and suggestions on improving this code
and in making it available on more platforms.


System Requirements:
====================
Operating system: OSX, Linux, Windows

Python Dependencies:
- Numpy
- OpenCV


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using
the HTML model. The code shows how to:
  - Load the HTML hand texture model
  - Edit texture parameters of the model to create a new hand texture
  - Save the resulting texture as a 2D rgb image mesh in .png format
  - The texture can be visualized by opening the ./vis.obj in MeshLab


'''


from utils.HTML_numpy import HTML_numpy
import cv2
import numpy as np
import os

if __name__ == '__main__':
    base_path = r"./TextureBasis"

    # Select model, with or W/O shading removal
    # model_path = os.path.join(base_path,'model_wosr', 'model.pkl')
    model_path = os.path.join(base_path, 'model_sr', 'model.pkl')

    # load model
    m = HTML_numpy(model_path)

    # play the parameter
    alpha = np.zeros(101)
    alpha[0:6] = np.array([2, -1, 1, 2, 3, -1])
    # alpha[0:6] = np.array([0, 0, 0, 0, 0, 0])

    # generate and store the texture
    tex = m.get_mano_texture(alpha)
    cv2.imwrite("./texture.png", (tex*255).astype(int)[..., ::-1])

