from torch import nn
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, BlendParams, HardFlatShader
)


class RgbRenderer(nn.Module):
    def __init__(self, H, W, device='cuda'):
        super().__init__()

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=3,
        )
        
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.shader = HardFlatShader(device=device)
        
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def forward(self, mesh, cameras):
        return self.renderer(mesh, cameras=cameras)


class MaskRenderer(nn.Module):
    def __init__(self, H, W, device='cuda'):
        super().__init__()

        raster_settings_mask = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings_mask)
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))

    def forward(self, mesh, cameras):
        fragments = self.rasterizer(mesh, cameras=cameras)
        return self.shader(fragments, mesh, cameras=cameras), fragments
