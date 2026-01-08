from pathlib import Path

import numpy as np
from PIL import Image
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from src.utils.preprocessing import load_mesh, load_poses, load_intrinsics, get_opencv_cameras_batch
from src.data.renderer import RgbRenderer, MaskRenderer


class ScenePreprocessor:
    """Class to produce and save views/masks for a given scene."""

    def __init__(self, replica_path, processed_path, scene_name, img_height, img_width, batch_size=5, device="cuda"):

        self.replica_path = Path(replica_path)
        self.processed_path = Path(processed_path)
        self.scene_name = scene_name
        self.device = device

        self.B = batch_size
        
        self.mesh = load_mesh(self.replica_path / scene_name / f'{scene_name}_mesh.ply', device)

        # not to build meshes_batch every time in get_batch
        self.meshes_batch = self._build_mesh_batch(self.B)
        
        self.face_to_instance = self._map_face_to_instance(self.mesh.faces_packed())

        self.poses = load_poses(self.replica_path / scene_name / "poses")
        
        # camera intrinsics
        self.K = load_intrinsics(self.replica_path / scene_name / "intrinsics.txt")
        self.H, self.W = img_height, img_width

        self.rgb_renderer = RgbRenderer(self.H, self.W, device)
        self.mask_renderer = MaskRenderer(self.H, self.W, device)


    @torch.no_grad()
    def get_batch(self, indices):
        camera_batch = get_opencv_cameras_batch(self.poses[indices], self.H, self.W, self.K, self.device)

        B = len(indices)
        if B != self.B:
            # rebuild mesh batch
            self.meshes_batch = self._build_mesh_batch(B)

        # render RGB and masks
        rgb_batch = self.rgb_renderer(self.meshes_batch, cameras=camera_batch)
        mask_batch, fragments_batch = self.mask_renderer(self.meshes_batch, cameras=camera_batch)

        # process outputs
        rgb_images = rgb_batch[..., :3]  # keep only RGB channels

        # map pixels to instance
        pix_to_instances_list = []
        faces_per_mesh = self.meshes_batch.faces_list()[0].shape[0]
        unique_instances_list = []
        
        for b in range(B):
            pix_to_face = fragments_batch.pix_to_face[b, ..., 0]  # packed
            valid = pix_to_face >= 0
        
            # convert packed → local face indices
            pix_to_face_local = pix_to_face.clone()
            pix_to_face_local[valid] -= b * faces_per_mesh
        
            # map to instance ids
            pix_to_instance = torch.full_like(pix_to_face, -1)
            pix_to_instance[valid] = self.face_to_instance[pix_to_face_local[valid]]
        
            pix_to_instances_list.append(pix_to_instance)

            unique_instances = pix_to_instance.unique(sorted=False)
            unique_instances = unique_instances[unique_instances != -1]
            unique_instances_list.append(unique_instances)
        
        pix_to_instances = torch.stack(pix_to_instances_list, dim=0)  # (B, H, W)

        return rgb_images, pix_to_instances, unique_instances_list


    def _build_mesh_batch(self, B):
        verts = self.mesh.verts_list()[0]      # get original verts tensor (V,3)
        faces = self.mesh.faces_list()[0]      # get original faces tensor (F,3)
        if self.mesh.textures is not None:
            verts_rgb = self.mesh.textures.verts_features_list()[0]  # (V,3)
            verts_rgb_batch = verts_rgb.unsqueeze(0).expand(B, -1, -1)  # (B,V,3)
            meshes_batch = Meshes(
                verts=[verts] * B,
                faces=[faces] * B,
                textures=TexturesVertex(verts_features=verts_rgb_batch)
            ).to(self.device)
        else:
            meshes_batch = Meshes(
                verts=[verts] * B,
                faces=[faces] * B
            ).to(self.device)
        return meshes_batch
        
    
    def _map_face_to_instance(self, faces):
        vert_to_instance = np.loadtxt(self.replica_path / "ground_truth" / f"{self.scene_name}.txt")
        vert_to_instance = torch.tensor(vert_to_instance, device=self.device).long()
        # lookup vertex labels → (F, 3)
        face_vertex_instances = vert_to_instance[faces]
        # majority vote per face
        face_to_instance = face_vertex_instances.mode(dim=1).values   # (F,)
        return face_to_instance


    def process_all_views(self, offset=0):
        N = len(self.poses)
        from math import ceil
        num_batches = ceil(N / self.B)
        
        for start in range(offset, N, self.B):
            end = min(start + self.B, N)
            indices = range(start, end)
        
            rgb_batch, inst_batch, unique_inst_batch = self.get_batch(indices)

            for b, idx in enumerate(indices):
                rgb = rgb_batch[b]      # batch-local index
                mask = inst_batch[b]    # batch-local index
                unique_instances = unique_inst_batch[b]
                self._save_rgb_tensor_and_mask(rgb, mask, unique_instances, idx)

            print(f"Processed batch {start // self.B}/{num_batches}.")


    def _save_rgb_tensor_and_mask(self, rgb_image, mask, unique_instances, index):
        path = self.processed_path / self.scene_name / str(index)
        path.mkdir(parents=True, exist_ok=True)
        
        img = rgb_image.detach().cpu()
        if img.dtype != torch.uint8:
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        img = img.numpy()
        Image.fromarray(img, mode="RGB").save(path / "rgb.png")
    
        np.save(path / "instance_mask.npy", mask.cpu().numpy())
        np.save(path / "unique_instances", unique_instances.cpu().numpy())


if __name__ == '__main__':
    import gc
    from src.configs.schema import BaseConfig
    from src.utils.setup_utils import load_config

    base_cfg = load_config(BaseConfig, "src/configs/base.json")
    
    scenes = base_cfg.scenes.train + base_cfg.scenes.eval
    for scene in scenes:
        print(f"Processing scene {scene}.")
        preprocessor = ScenePreprocessor(
            base_cfg.paths.raw_replica,
            base_cfg.paths.processed_root,
            scene,
            base_cfg.camera.height,
            base_cfg.camera.width
        )
        preprocessor.process_all_views()
        # clear memory and cache
        del preprocessor
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Processed whole dataset.")