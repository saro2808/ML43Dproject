import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import trimesh
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes


def load_intrinsics(path):
    K = np.loadtxt(path)[:3,:3]
    return K


def load_poses(pose_dir):
    pose_files = sorted(glob.glob(str(pose_dir) + "/*.txt"))
    poses = [np.loadtxt(f) for f in pose_files]
    return poses


def load_images(image_dir):
    image_files = sorted(glob.glob(str(image_dir) + "/*.jpg"))
    images = [np.array(Image.open(f)) for f in image_files]
    return images


def build_camera_from_replica(K, pose, H, W, device='cuda'):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # cy = H - 1 - cy

    Rcw = pose[:3,:3]
    tcw = pose[:3,3]

    # R=Rcw
    # T=tcw
    R = Rcw.T
    T = -R @ tcw

    # cv2_to_p3d = np.array([
    #     [1,  0,  0],
    #     [0, -1,  0],
    #     [0,  0, -1]
    # ], dtype=np.float32)
    
    # R = cv2_to_p3d @ R
    # T = cv2_to_p3d @ T

    return PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        R=torch.tensor(R)[None].float().to(device),
        T=torch.tensor(T)[None].float().to(device),
        device=device,
        in_ndc=False        # important!
    )


def load_mesh(mesh_path, device='cuda'):
    # Load with trimesh (handles polygonal faces)
    mesh_tm = trimesh.load(mesh_path, process=False)
    
    # vertices and faces
    verts_pos = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
    faces_idx = torch.tensor(mesh_tm.faces, dtype=torch.long)
    
    # vertex colors (if available)
    if hasattr(mesh_tm.visual, "vertex_colors") and mesh_tm.visual.vertex_colors is not None:
        verts_rgb = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32, device=device) / 255.0
    else:
        verts_rgb = torch.ones_like(verts_pos, device=device)  # fallback to white
    
    # Create PyTorch3D mesh
    return Meshes(
        verts=[verts_pos],
        faces=[faces_idx],
        textures=TexturesVertex(verts_features=verts_rgb[None])
    ).to(device)


def plot_pair(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    ax[0].imshow(img1)
    # ax[0].set_title("First Image")
    ax[0].axis('off') # Optional: hide axis ticks
    
    # Display the second image
    ax[1].imshow(img2)
    # ax[1].set_title("Second Image")
    ax[1].axis('off')
    
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()