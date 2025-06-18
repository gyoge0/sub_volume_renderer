import pygfx.renderers.wgpu as wgpu
from jinja2 import PackageLoader
from pygfx.renderers.wgpu import (
    BaseShader,
    load_wgsl,
    register_wgsl_loader,
    to_texture_format,
)

# volumeshader internally imports wgpu.ShaderStage to construct the shader stages,
# but wgpu.ShaderStage doesn't exist within pygfx.wgpu for us.
# to get around this, we import the constructed flag directly from volumeshader.
# alternative would be importing from the wgpu module (and not gfx.renderers.wgpu) or
# hard coding the flag.
from pygfx.renderers.wgpu.shaders.volumeshader import vertex_and_fragment

from ._wobject import GlobalSparseVolume, GlobalSparseVolumeMaterial

register_wgsl_loader("rbvr", PackageLoader("rbvr", "shaders"))


@wgpu.register_wgpu_render_function(GlobalSparseVolume, GlobalSparseVolumeMaterial)
class GlobalSparseVolumeShader(wgpu.shaders.volumeshader.VolumeRayShader):
    def __init__(self, wobject: GlobalSparseVolume, **kwargs):
        # skip the BaseVolumeShader init because we don't have a geometry.grid and
        # that breaks the init.
        # we perform all the actions from BaseVolumeShader inside this init.
        BaseShader.__init__(self, wobject, **kwargs)

        material: GlobalSparseVolumeMaterial = wobject.material

        # BaseVolumeShader makes a bunch of assertions here about the geometry,
        # but since we require wobject to be a CustomVolume,
        # the wobject will have already made those assertions for us.

        # Set the render mode
        # This should always be "mip"
        # Could maybe just remove the parameter from the template
        self["mode"] = "mip"
        # Set image format
        self["climcorrection"] = ""
        fmt = to_texture_format(wobject.cache_texture.format)
        if "norm" in fmt or "float" in fmt:
            self["img_format"] = "f32"
            if "unorm" in fmt:
                self["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                self["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            self["img_format"] = "u32"
        else:
            self["img_format"] = "i32"

        # Set gamma
        self["gamma"] = material.gamma

        # Channels
        self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        # Colorspace
        self["colorspace"] = wobject.cache_texture.colorspace
        if material.map is not None:
            self["colorspace"] = material.map.texture.colorspace

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = [
            wgpu.Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            wgpu.Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            wgpu.Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # our indirection texture
        # we don't provide a sampler for the indirection texture since the shader
        # is expected to only read valid texels from it
        t_indirection = wgpu.GfxTextureView(wobject.indirection_texture)
        bindings.append(
            wgpu.Binding(
                "t_indirection", "texture/auto", t_indirection, vertex_and_fragment
            )
        )

        # our cache texture
        # we provide a sampler for the cache texture to allow filtering
        t_cache = wgpu.GfxTextureView(wobject.cache_texture)
        s_cache = wgpu.GfxSampler(
            # material.interpolation,
            "nearest",
            "clamp",
        )
        bindings.append(
            wgpu.Binding("s_cache", "sampler/filtering", s_cache, "FRAGMENT")
        )
        bindings.append(
            wgpu.Binding("t_cache", "texture/auto", t_cache, vertex_and_fragment)
        )

        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_code(self):
        return load_wgsl(
            "ring_buffer_volume_renderer.wgsl", package_name="rbvr.shaders"
        )
