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
- pytorch
- pytorch3D == 0.1


About the Script:
=================
This script demonstrates how to generate and render 3D textured hand mesh
with our HTML and MANO model to help users get started with using the HTML
model. The code shows how to:
  - Apply the HTML hand texture model to the MANO hand model
  - Edit pose, shape, texture parameters of the model to create a new 3D hand
  - Render the 3D hand mesh by a differentialable renderer Pytorch3D
  - Save the resulting texture as a 2D rgb image mesh in .png format
  - Save the rendered hand image as a 2D rgb image mesh in .png format
  - The texture can be visualized by opening the ./vis.obj in MeshLab

Note:
  - This script requires the ./MANO_RIGHT.pkl .
  Download the MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/
  - The UV coordinators (./TextureBasis/uvs.pkl) is only for the MANO right hand mesh
  For the left hand, the col of faces_uvs may need to be swapped
'''
import random

import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pytorch3d.structures import Meshes#, Textures
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, Materials, TexturesVertex, Textures, SoftPhongShader, TexturesUV
)

# from pytorch3d.renderer import (
#     OpenGLPerspectiveCameras, look_at_view_transform,
#     RasterizationSettings, MeshRenderer, MeshRasterizer,
#     PointLights, Materials,
# )


from utils.HTML import MANO_SMPL_HTML
from yy_try import render_img, render_img_rotate, render_cubes

class MANO_Renderer: #Abstract Base Class. Don't use this!
    def __init__(self, model_path, tex_path, uvs_path ,device):
        self.html = MANO_SMPL_HTML(model_path, tex_path, uvs_path)

        self.verts_uvs = torch.unsqueeze(self.html.verts_uvs, 0)#.cuda()
        self.faces_uvs= torch.unsqueeze(self.html.faces_uvs, 0)#.cuda()
        self.faces_idx = torch.unsqueeze(self.html.faces_idx, 0)#.cuda()

        self.cameras = None
        self.image_size = None
        self.renderer = None

        self.device = device


    def config_renderer(self):
        lights = PointLights(device=self.device, location=[[1.0, 1.0, -2.0]], ambient_color=[[1.0, 1.0, 1.0]],
                             diffuse_color=[[0., 0., 0.]], specular_color=[[0, 0, 0]])
        material = Materials(device=self.device, ambient_color=[[1, 1, 1]], diffuse_color=[[1, 1, 1]], specular_color=[[0, 0, 0]])

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=0
        )



        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=lights,
                materials=material,
                cameras=self.cameras)
        )

        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=self.cameras,
        #         raster_settings=raster_settings
        #     ),
        #     shader=TexturedPhongShader(
        #         device=self.device,
        #         lights=lights,
        #         materials = material,
        #         cameras = self.cameras)
        # )
        
        return renderer
        
    def render_hand(self, pose, shape, tex_param, angle=0, axis="Z", is_dorsal=True):
        batch_size = pose.shape[0]  # yy: pose shape [bs, 45]

        verts_uvs = self.verts_uvs.repeat((batch_size, 1, 1)).contiguous()
        faces_idx = self.faces_idx.repeat((batch_size, 1, 1)).contiguous()
        faces_uvs = self.faces_uvs.repeat((batch_size, 1, 1)).contiguous()

        # yy: original texture
        new_tex_img = self.html.get_mano_texture(tex_param)
        new_tex_img = new_tex_img.contiguous()

        # yy: Load edited texture
        new_tex_img_edited = cv2.imread('./blur_edited_texture_v3.png')
        new_tex_img_edited = TF.to_tensor(new_tex_img_edited)
        new_tex_img_edited.unsqueeze_(0)
        new_tex_img_edited = torch.transpose(torch.transpose(new_tex_img_edited, 1, 3), 1, 2)

        # MANO geometry's rotation, we set to all 0 in our demo.
        rotation = torch.tensor([[0, 0, 0]], dtype=torch.float32)#.cuda()
        # scale and global translation of the MANO goemetry, we also set to all 0.
        scale_and_trans = torch.tensor([[1, 0, 0, 0]], dtype=torch.float)#.cuda()


        # yy: @Atith, here's where keypoints generated (i.e., the joints here).
        #  They do a scale (x1000) in self.html.get_mano_vertices, so I divide by 1000 later.
        #  This might be the issue but I'm not sure. Maybe you can check this self.html.get_mano_vertices.
        vertices, joints = self.html.get_mano_vertices(rotation, pose, shape, scale_and_trans)
        jts = joints.clone()
        jts /= 1000
        jts = jts - torch.mean(vertices, 1)
        # move vertices to the center
        verts = vertices.clone()
        verts = verts - torch.mean(verts, 1)  # yy: shape [1, 778, 3]

        # yy: my renderer
        # yy: old
        # images, image_cubes = render_img(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img, joints / 1000, is_dorsal=is_dorsal)
        # yy: new -> rotation
        images_ori, _, _ = render_img_rotate(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img, jts, angle, axis)
        images_edited_texture, _, _ = render_img_rotate(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img_edited, jts,
                                                    angle, axis, is_bicolor=True)
        image_cubes, kp = render_cubes(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img, jts, angle, axis)


        # yy: the original one
        # tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=new_tex_img)
        # tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=new_tex_img)
        # mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)
        #
        # images = self.renderer(mesh)

        return images_ori, images_edited_texture, image_cubes, vertices, new_tex_img, kp

    def config_camera(self, image_size):
        # config the camera parameter, e.g. position and rotation.
        R, T = look_at_view_transform(0.4, elev = -89, azim= 0)
        self.cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
        self.image_size = image_size
        self.renderer = self.config_renderer()



def gen_rand_params( len, range_scale):
    rand_val= range_scale*(torch.rand(len) - 0.5)
    return rand_val






# yy: old version -> only generate dorsal or ventral instead of rotation way
# if __name__ == '__main__':
#     # device = torch.device("cuda:0")
#     device = torch.device("cpu")
#     # torch.cuda.set_device(device)
#     base_path = "./"
#
#     hand_type = 'left' #'left'
#
#     # to run this script, the MANO_RIGHT.pkl should be in the folder.
#     # download the MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/
#     # our predefined uv coordinator and uv face index for MANO_RIGHT hand mesh
#
#     if hand_type == 'right':
#         model_path = os.path.join(base_path, "MANO_RIGHT.pkl")
#         uv_path = os.path.join(base_path, "TextureBasis/uvs_right.pkl")
#     elif hand_type == 'left':
#         model_path = os.path.join(base_path, "MANO_LEFT.pkl")
#         uv_path = os.path.join(base_path, "TextureBasis/uvs_left.pkl")
#
#     # select the texture model, either model_sr or model_wosr
#     tex_path = os.path.join(base_path, "TextureBasis/model_sr/model.pkl")
#     renderer = MANO_Renderer(model_path, tex_path, uv_path, device)
#     renderer.config_camera(image_size=512)
#
#     # yy: iterate to get more images
#     # For dorsal
#     for i in range(12):
#         is_dorsal = True
#         print(i)
#         # generate rand parameters for pose, shape and texture
#         pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)#.cuda()
#         shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)#.cuda()
#         tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)#.cuda()
#
#         images, image_cubes, vertices, new_tex_img, joints = renderer.render_hand(pose_param, shape_param, tex_param, is_dorsal=is_dorsal)
#         cv2.imwrite("debug_imgs/more_poses/dorsal/rendered_{}.png".format(i), (images[0, ..., :3].cpu().numpy()*255).astype(int)[..., ::-1])
#         cv2.imwrite("debug_imgs/more_poses/dorsal/texture_{}.png".format(i), (new_tex_img[0, ..., :3].cpu().numpy()*255).astype(int)[..., ::-1])
#         cv2.imwrite("debug_imgs/more_poses/dorsal/cubes_{}.png".format(i),
#                     (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
#         # yy: @Atith, here's where to save the keypoints (21, 3). I haven't verified the correctness of the kp, you can verify it first.
#         np.save("debug_imgs/kp/dorsal/kp_{}.npy".format(i), np.squeeze(joints))
#         a=1
#
#     # For ventral
#     for i in range(12):
#         is_dorsal = False
#         print(i)
#         # generate rand parameters for pose, shape and texture
#         pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)#.cuda()
#         shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)#.cuda()
#         tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)#.cuda()
#
#         images, image_cubes, vertices, new_tex_img, joints = renderer.render_hand(pose_param, shape_param, tex_param, is_dorsal=is_dorsal)
#         cv2.imwrite("debug_imgs/more_poses/ventral/rendered_{}.png".format(i), (images[0, ..., :3].cpu().numpy()*255).astype(int)[..., ::-1])
#         cv2.imwrite("debug_imgs/more_poses/ventral/texture_{}.png".format(i), (new_tex_img[0, ..., :3].cpu().numpy()*255).astype(int)[..., ::-1])
#         cv2.imwrite("debug_imgs/more_poses/ventral/cubes_{}.png".format(i),
#                     (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
#         # yy: @Atith, here's where to save the keypoints (21, 3). I haven't verified the correctness of the kp, you can verify it first.
#         np.save("debug_imgs/kp/ventral/kp_{}.npy".format(i), np.squeeze(joints))







# yy: new rotation way
if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # torch.cuda.set_device(device)
    base_path = "./"


    # if random.random() > 0.5:
    #     hand_type = 'left'
    #     print("left")
    # else:
    #     hand_type = 'right'
    #     print("right")
    #
    # # to run this script, the MANO_RIGHT.pkl should be in the folder.
    # # download the MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/
    # # our predefined uv coordinator and uv face index for MANO_RIGHT hand mesh
    #
    # if hand_type == 'right':
    #     model_path = os.path.join(base_path, "MANO_RIGHT.pkl")
    #     uv_path = os.path.join(base_path, "TextureBasis/uvs_right.pkl")
    # elif hand_type == 'left':
    #     model_path = os.path.join(base_path, "MANO_LEFT.pkl")
    #     uv_path = os.path.join(base_path, "TextureBasis/uvs_left.pkl")
    #
    #
    # # # Add wrist
    # # model_path = os.path.join(base_path, "SMPLH_male.pkl")
    # # uv_path = os.path.join(base_path, "TextureBasis/uvs_left.pkl")
    #
    #
    # # select the texture model, either model_sr or model_wosr
    # tex_path = os.path.join(base_path, "TextureBasis/model_sr/model.pkl")
    # renderer = MANO_Renderer(model_path, tex_path, uv_path, device)
    # renderer.config_camera(image_size=256)

    # yy: iterate to get more images
    """
    angle: 0 -> ventral, 180 -> dorsal
    so, 
    - ventral range set as [-60, 60]
    - dorsal range set as [120, 240]
    """
    # # yy: for test
    # for angle in [0, 90, 180, 270]:
    #     pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)  # .cuda()
    #     shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)  # .cuda()
    #     tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)  # .cuda()
    #     axis = "X"
    #     images, image_cubes, vertices, new_tex_img, joints = renderer.render_hand(pose_param, shape_param, tex_param, angle, axis)
    #     cv2.imwrite("debug_imgs/debug_rotation/rendered_{}.png".format(angle),
    #                 (images[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("debug_imgs/debug_rotation/texture_{}.png".format(angle),
    #                 (new_tex_img[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("debug_imgs/debug_rotation/cubes_{}.png".format(angle),
    #                 (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
    #     a = 1

    # # yy: rotation angles (old -> have both dorsal & ventral && the rotation angles & axis is restricted)
    # ventral_range, dorsal_range = np.arange(-60, 60, 1), np.arange(120, 240, 1)
    # # ventral
    # print(">> Ventral")
    # for i in range(5):
    #     pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)  # .cuda()
    #     shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)  # .cuda()
    #     tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)  # .cuda()
    #     angle = np.random.choice(ventral_range, 1)[0]
    #     print(angle)
    #     axis = "X"
    #     images_ori, images_edited_texture, image_cubes, vertices, new_tex_img, joints = renderer.render_hand(pose_param, shape_param, tex_param, angle, axis)
    #     cv2.imwrite("final_imgs/ventral/rendered_hand_{}.png".format(str(i).zfill(5)),
    #                 (images_ori[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/ventral/edited_texture_hand_{}.png".format(str(i).zfill(5)),
    #                 (images_edited_texture[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/ventral/texture_{}.png".format(str(i).zfill(5)),
    #                 (new_tex_img[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/ventral/cubes_{}.png".format(str(i).zfill(5)),
    #                 (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
    #     np.save("final_imgs/ventral/kp_{}.npy".format(str(i).zfill(5)), np.squeeze(joints))
    #
    # # dorsal
    # print(">> Dorsal")
    # for i in range(5):
    #     pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)  # .cuda()
    #     shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)  # .cuda()
    #     tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)  # .cuda()
    #     angle = np.random.choice(dorsal_range, 1)[0]
    #     print(angle)
    #     axis = "X"
    #     images_ori, images_edited_texture, image_cubes, vertices, new_tex_img, joints = renderer.render_hand(pose_param, shape_param, tex_param, angle, axis)
    #     cv2.imwrite("final_imgs/dorsal/rendered_hand_{}.png".format(str(i).zfill(5)),
    #                 (images_ori[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/dorsal/edited_texture_hand_{}.png".format(str(i).zfill(5)),
    #                 (images_edited_texture[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/dorsal/texture_{}.png".format(str(i).zfill(5)),
    #                 (new_tex_img[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
    #     cv2.imwrite("final_imgs/dorsal/cubes_{}.png".format(str(i).zfill(5)),
    #                 (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
    #     np.save("final_imgs/dorsal/kp_{}.npy".format(str(i).zfill(5)), np.squeeze(joints))

    dir_name = 'dataset_10000_256_left_right_no_wrists_fix_angle'
    dir_ls = ['{}/rendered_hand'.format(dir_name), '{}/bicolor_hand'.format(dir_name),
              '{}/texture'.format(dir_name), '{}/cubes'.format(dir_name), '{}/kp'.format(dir_name)]
    for dir_pth in dir_ls:
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)

    DATASET_SIZE = 10
    angle_range = np.arange(-90, 270, 1)
    angle_range = np.array([0])

    folder_name = dir_name
    # folder_name = "wrist_test"
    start_time = time.time()
    for i in range(DATASET_SIZE):
        if random.random() > 0.5:
            hand_type = 'left'
            print("left")
        else:
            hand_type = 'right'
            print("right")

        if hand_type == 'right':
            model_path = os.path.join(base_path, "MANO_RIGHT.pkl")
            uv_path = os.path.join(base_path, "TextureBasis/uvs_right.pkl")
        elif hand_type == 'left':
            model_path = os.path.join(base_path, "MANO_LEFT.pkl")
            uv_path = os.path.join(base_path, "TextureBasis/uvs_left.pkl")

        tex_path = os.path.join(base_path, "TextureBasis/model_sr/model.pkl")
        renderer = MANO_Renderer(model_path, tex_path, uv_path, device)
        renderer.config_camera(image_size=256)


        pose_param = torch.unsqueeze(gen_rand_params(45, 3), 0)  # .cuda()
        shape_param = torch.unsqueeze(gen_rand_params(10, 3), 0)  # .cuda()
        tex_param = torch.unsqueeze(gen_rand_params(101, 3), 0)  # .cuda()
        while True:
            angle = np.random.choice(angle_range, 1)[0]
            axis = np.random.choice(["X", "Y", "Z"], 1)[0]
            # if not (axis == "Z" and (((angle >= -90) and (angle <= -45)) or ((angle >= 200) and (angle <= 270)))):
            if not axis == "Z":
                break
        # axis = "X"  # yy: select X if we only wanna rotate around X axis
        print(i, axis, angle)
        images_ori, images_edited_texture, image_cubes, vertices, new_tex_img, joints = \
            renderer.render_hand(pose_param, shape_param, tex_param, angle, axis)
        cv2.imwrite("{}/rendered_hand/{}_{}.png".format(folder_name, str(i).zfill(5), hand_type),
                    (images_ori[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
        cv2.imwrite("{}/bicolor_hand/{}_{}.png".format(folder_name, str(i).zfill(5), hand_type),
                    (images_edited_texture[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
        cv2.imwrite("{}/texture/{}_{}.png".format(folder_name, str(i).zfill(5), hand_type),
                    (new_tex_img[0, ..., :3].cpu().numpy() * 255).astype(int)[..., ::-1])
        cv2.imwrite("{}/cubes/{}_{}.png".format(folder_name, str(i).zfill(5), hand_type),
                    (image_cubes[0, ..., :3].cpu().detach().numpy() * 255).astype(int)[..., ::-1])
        np.save("{}/kp/{}_{}.npy".format(folder_name, str(i).zfill(5), hand_type), np.squeeze(joints))
        if i % 10 == 0:
            print("time elapsed: ", time.time() - start_time)
            start_time = time.time()
