import glob

import numpy as np
from PIL import Image
import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


def load_intrinsics(path):
    K = np.loadtxt(path)[:3,:3]
    return K


def load_poses(pose_dir):
    pose_files = sorted(glob.glob(str(pose_dir) + "/*.txt"))
    poses = [np.loadtxt(f) for f in pose_files]
    return np.array(poses)


def load_images(image_dir):
    image_files = sorted(glob.glob(str(image_dir) + "/*.jpg"))
    images = [np.array(Image.open(f)) for f in image_files]
    return images


def get_opencv_cameras_batch(poses, img_height, img_width, intrinsic_mat, device="cuda"):
    R = torch.tensor(poses[:, :3, :3], dtype=torch.float32)
    T = torch.tensor(poses[:, :3, 3], dtype=torch.float32).unsqueeze(-1)  # (B,3,1)

    R = R.transpose(1, 2)
    T = -torch.bmm(R, T).squeeze(-1)   # back to (B,3)

    bsize = R.shape[0]

    # create camera with opencv function
    image_size = torch.Tensor((img_height, img_width))
    image_size_repeat = torch.tile(image_size.reshape(-1, 2), (bsize, 1))
    intrinsic_repeat = torch.Tensor(intrinsic_mat).unsqueeze(0).expand(bsize, -1, -1)
    
    opencv_cameras = cameras_from_opencv_projection(
        R=R.to(device),  # N, 3, 3
        tvec=T.to(device),  # N, 3
        camera_matrix=intrinsic_repeat.to(device),  # N, 3, 3
        image_size=image_size_repeat.to(device)  # N, 2 h,w
    )
    return opencv_cameras


def load_mesh(mesh_path, device='cuda'):
    from trimesh import load
    mesh_tm = load(mesh_path, process=False)
    
    verts_pos = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
    faces_idx = torch.tensor(mesh_tm.faces, dtype=torch.long)
    
    # vertex colors (if available)
    if hasattr(mesh_tm.visual, "vertex_colors") and mesh_tm.visual.vertex_colors is not None:
        verts_rgb = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32, device=device) / 255.0
    else:
        verts_rgb = torch.ones_like(verts_pos, device=device)  # fallback to white
    
    return Meshes(
        verts=[verts_pos],
        faces=[faces_idx],
        textures=TexturesVertex(verts_features=verts_rgb[None])
    ).to(device)