from pathlib import Path

import numpy as np
import torch

from utils import load_mesh, load_poses, load_images, load_intrinsics, get_opencv_cameras_batch
from renderer import RgbRenderer, MaskRenderer


root_path = Path("data/replica")


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a given scene"""

    def __init__(self, scene_name, device="cuda"):

        self.scene_name = scene_name
        self.device = device
        
        self.mesh = load_mesh(root_path / scene_name / f'{scene_name}_mesh.ply', device)
        self.face_to_instance = self.map_face_to_instance()

        poses = load_poses(root_path / scene_name / "poses")
        # colorful images
        self.images = load_images(root_path / scene_name / "color")
        assert len(poses) == len(self.images), "Number of poses should equal number of images."
        
        # camera intrinsics
        K = load_intrinsics(root_path / scene_name / "intrinsics.txt")
        H, W, _ = self.images[0].shape  # (360, 640, 3)
        
        self.cameras = [
            get_opencv_cameras_batch(pose[None], H, W, K).to(device) for pose in poses
        ]

        self.rgb_renderer = RgbRenderer(H, W, device)
        self.mask_renderer = MaskRenderer(H, W, device)

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        rgb_image = self.rgb_renderer(self.mesh, cameras=self.cameras[idx])
        rgb_image = rgb_image[0, ..., :3]  # HxWx3
        
        mask, fragments = self.mask_renderer(self.mesh, cameras=self.cameras[idx])
        mask = mask[0, ..., 3]  # HxW

        pix_to_face = fragments.pix_to_face[..., 0]
        pix_to_instance = self.map_pix_to_instance(pix_to_face)

        return rgb_image, pix_to_instance

    
    def map_face_to_instance(self):
        vert_to_instance = np.loadtxt(root_path / "ground_truth" / f"{self.scene_name}.txt")
        vert_to_instance = torch.tensor(vert_to_instance, device=self.device).long()
        
        faces = self.mesh.faces_packed()  # (F, 3)
        # lookup vertex labels → (F, 3)
        face_vertex_instances = vert_to_instance[faces]
        # majority vote per face
        face_to_instance = face_vertex_instances.mode(dim=1).values   # (F,)
        
        return face_to_instance

    
    def map_pix_to_instance(self, pix_to_face):
        pix_to_instance = torch.full_like(pix_to_face, fill_value=-1)
    
        valid = pix_to_face >= 0
        pix_to_instance[valid] = self.face_to_instance[pix_to_face[valid]]
        
        return pix_to_instance
