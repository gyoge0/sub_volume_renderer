// this file is modified from the original
// a lot of functions have been split up into many of the other wgsl files
// https://github.com/pygfx/pygfx/blob/769dc54a44bcf2f31fbd9f9276171890c11e57f7/pygfx/renderers/wgpu/wgsl/volume_ray.wgsl

// Volume rendering via raycasting. Multiple modes supported.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
$$ if mode == 'iso'
    {$ include 'pygfx.light_phong_simple.wgsl' $}
$$ endif
{$ include 'sub_volume.volume_common.wgsl' $}

{$ include 'sub_volume.vs_main.wgsl' $}
{$ include 'sub_volume.raycast.wgsl' $}
{$ include 'sub_volume.fs_main.wgsl' $}
