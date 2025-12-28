from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, BlendParams
)


class RgbRenderer:
    
    def __init__(self, H, W, device='cuda'):
        
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=20,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=device)
        )

    def __call__(self, mesh, cameras):
        return self.renderer(mesh, cameras=cameras)


class MaskRenderer:

    def __init__(self, H, W, device='cuda'):

        raster_settings_mask = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=50,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_mask),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
        )

    def __call__(self, mesh, cameras):
        return self.renderer(mesh, cameras=cameras)
