License:
========
To learn about HTML, please visit our website: http://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/
You can find the HTML paper at: https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/content/HandTextureModel_ECCV2020.pdf


For comments or questions, please email us at: Jiayi Wang
					      jwang@mpi-inf.mpg.de


System Requirements:
====================
Operating system: OSX, Linux, Windows

Python Dependencies:
- Numpy 
- Pytorch		 
- OpenCV 		  
- Pytorch3D

The MANO model file:
MANO_RIGHT.pkl   (From: https://mano.is.tue.mpg.de/. Place this under project directory)


Installation (Written by yy, other sections in readme are written by the author of the codebase)
=========================
First, download the two folders:
- TextureBasis: https://www.dropbox.com/scl/fo/05zuck7rqkyg0ovk229g2/h?dl=0&rlkey=wp4wvi110z86zvz1p6a979hgc
- TextureSet: https://www.dropbox.com/scl/fo/a8bw4z8z30gk7kbkrcn42/h?dl=0&rlkey=6hbevcm8u1azwc0ldtdi11wsx

Then, `git clone https://github.com/YY-GX/Hand-Synthesis.git` to download the `Hand-Synthesis`. Put the two folders (i.e., TextureBasis & TextureSet) under the `Hand-Synthesis` folder.

This repo is outdated, so please stick to the following installation sequence (test on Ubuntu20):
```shell
# Create Conda Env (py 3.7 is required)
conda create --name html python=3.7 -y

# Install torch
pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html

# Please overlook this part!
## Install pytorch3d: Go https://github.com/facebookresearch/pytorch3d/releases/tag/v0.1.0 
## to install the 1.0.0 version first. Unzip it.
#cd pytorch3d-0.1.0
#pip install -e .

# Install pytorch3d
pip install pytorch3d

# Install other packages
pip install imageio
pip install opencv-python

```
You may also need to install some other libraries, just `pip install` those that are notified "Not Found". 



Getting Started:
================

1. Run the Hello World scripts:
-------------------------------

> python HTML__hello_world.py

This script only require Numpy.
HTML__hello_world.py script generate a hand texture from our hand model.
User can select to either use the shading removal or no shading removal texutre model.
The generated texture can be visualized by importing the vis.obj in MeshLab. 

2. Run the rendering scripts:
-------------------------------

> python Render_HTML.py

This script use pytorch3D to render our HTML hand texture model on a MANO hand mesh.
It require the MANO model file (MANO_RIGHT.pkl) to be first downloaded.
It will output the rendered hand image.

Note:
The HTML__hello_world script only require Numpy. The scripts are provided as a sample to help you get started. 

Acknowledgements:
The code is based on the release code of ICCV Hands2019 Challenge (https://sites.google.com/view/hands2019/challenge)
Git Repo: https://github.com/anilarmagan/HANDS19-Challenge-Toolbox
Therefore, we would like to kindly thank Dr. Anil Armagan.
