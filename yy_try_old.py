import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch3d.transforms import Translate, RotateAxisAngle
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import numpy as np


# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))


# yy: function used by Render_HTML.py, used for rendering images (@Atith, overlook this function)
def render_img(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img, kp, is_dorsal=True):
    tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=new_tex_img)
    mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)


    # yy: @Atith, here's where to rotate the camera (but we wanna rotate the mesh)
    if is_dorsal:
        R, T = look_at_view_transform(0.2, 270, 90)
    else:
        R, T = look_at_view_transform(0.2, 90, 90)

    # yy: insert rotating mesh code here
    pass

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]], specular_color=((0.0, 0.0, 0.0),))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    images = renderer(mesh)
    # plt.figure(figsize=(20, 20))
    # # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.savefig('myrender.png')

    # yy: add cubes
    cubes_mesh = create_cubes_and_relocate(torch.squeeze(kp))
    renderer_cubes = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    image_cubes = renderer_cubes(cubes_mesh)



    return images, image_cubes



# yy: function used by Render_HTML.py, used for rendering images
#  @Atith, here's where I generate hand mesh & cubes meshes, can you check issues here?
def render_img_rotate(verts, faces_idx, faces_uvs, verts_uvs, new_tex_img, kp, angle, axis):
    # Set environment

    # Set R & T for camera (i.e., set camera position)
    R, T = look_at_view_transform(0.2, 90, 90)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]], specular_color=((0.0, 0.0, 0.0),), diffuse_color=((0.7, 0.7, 0.7),))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )







    # Create meshes (hand & cubes)

    # yy: Create hand mesh
    tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=new_tex_img)  # hand texture
    mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)  # This is hand mesh
    # rotate hand mesh
    mesh = rotate_mesh(mesh, angle, axis)
    # render hand mesh
    images = renderer(mesh)

    # yy: Create cubes mesh
    cubes_mesh = create_cubes_and_relocate(torch.squeeze(kp))
    # rotate cubes mesh
    cubes_mesh = rotate_mesh(cubes_mesh, angle, axis)
    # render cubes mesh
    image_cubes = renderer(cubes_mesh)



    return images, image_cubes


# yy: create cubes and translate them to positions of keypoints.
#  @Atith, this is where I add cubes, can you check the issues here?
def create_cubes_and_relocate(kp, size_ratio=0.001):
    """
    Create cubes and then translate them to positions of kps
    :param
    - kp: Torch [21, 3]
    - size_ratio： float

    :return: mesh
    """

    v_ls, f_ls = [], []
    for i in range(21):
        # verts & faces for one cube
        vertices = torch.tensor([[1., 0, 1.], [1., 0., -1.], [-1, 0, -1], [-1, 0, 1],
                                 [1, 2, 1], [1, 2, -1], [-1., 2., -1.], [-1., 2., 1.]]) * size_ratio

        faces = torch.tensor([[1, 0, 2], [0, 3, 2],
                              [2, 3, 6], [7, 6, 3],
                              [1, 4, 0], [1, 5, 4],
                              [1, 2, 6], [1, 6, 5],
                              [0, 4, 7], [0, 7, 3],
                              [4, 5, 7], [4, 6, 7],
                              ]) + 8 * i
        # position of one keypoint
        kp_pos = kp[i:i+1, :]

        # translate positions of cubes to keypoint's position
        trans = Translate(torch.nn.Parameter(kp_pos), device=torch.device("cpu"))
        tverts = trans.transform_points(vertices)

        v_ls.append(torch.unsqueeze(tverts, 0)), f_ls.append(torch.unsqueeze(faces, 0))


    # To crate a mesh with new verts & faces
    vs = torch.cat(v_ls, 1)
    fs = torch.cat(f_ls, 1)

    verts_rgb = torch.zeros((1, 8 * 21, 3))
    textures = Textures(verts_rgb=verts_rgb)

    mesh = Meshes(verts=vs, faces=fs, textures=textures)

    return mesh



# yy: rotate mesh
def rotate_mesh(mesh, angle, axis):
    """
    rotate mesh
    :param mesh: mesh
    :param angle: rotate angle
    :param axis: rotate which axis -> ["x", "Y", "Z"]
    :return:
    """
    rot = RotateAxisAngle(angle, axis)
    rverts = rot.transform_points(mesh.verts_list()[0])
    faces = mesh.faces_list()[0]
    mesh = Meshes(verts=torch.unsqueeze(rverts, 0), faces=torch.unsqueeze(faces, 0), textures=mesh.textures)
    return mesh






# yy: try with only one pose. @Atith, you can overlook the following parts

# for dist in np.arange(0.05, 0.65, 0.05):
#     for z_n in np.arange(0.2, 1.2, 0.2):
# for dist in [0.15, 0.2, 0.25, 0.3]:
#     for z_n in [0.1, 0.15, 0.2]:

# for y in [0, 90, 180, 270]:
#     for z in [0, 90, 180, 270]:

for y in [90]:
    for z in [90]:

        # Setup
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # Set paths
        DATA_DIR = "./data"
        # obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
        #
        # # Load obj file
        # mesh = load_objs_as_meshes([obj_filename], device=device)
        verts = torch.load("verts.pt")
        faces_idx = torch.load("faces_idx.pt")
        faces_uvs = torch.load("faces_uvs.pt")
        verts_uvs = torch.load("verts_uvs.pt")
        new_tex_img = torch.load("new_tex_img.pt")
        tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=new_tex_img)
        mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)

        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
        cm_dist = 0.3  # 0.6 looks good
        ag_y, ag_z = y, z
        R, T = look_at_view_transform(cm_dist, ag_y, ag_z)
        znear = 0.1
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        aa_factor = 5
        raster_settings = RasterizationSettings(
            image_size=512 * aa_factor,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        # yy: tune light here
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]], specular_color=((0.0, 0.0, 0.0),))

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        images = renderer(mesh)
        images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
        images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        plt.figure(figsize=(5.12, 5.12))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig('debug_imgs/final/img_y{}_z{}_lighttest.png'.format(ag_y, ag_z))