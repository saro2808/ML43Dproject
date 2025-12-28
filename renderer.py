from torch import nn
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, BlendParams
)


class RgbRenderer(nn.Module):
    def __init__(self, H, W, device='cuda'):
        super().__init__()

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=20,
        )
        
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.shader = SoftPhongShader(device=device)
        
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
            faces_per_pixel=50,
        )

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings_mask)
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
        
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def forward(self, mesh, cameras):
        return self.renderer(mesh, cameras=cameras)
