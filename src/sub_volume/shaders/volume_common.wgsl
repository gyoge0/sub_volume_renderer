// this file is modified from the original
// the `sample_vol` function has been removed!
// https://github.com/pygfx/pygfx/blob/769dc54a44bcf2f31fbd9f9276171890c11e57f7/pygfx/renderers/wgpu/wgsl/volume_common.wgsl


// Common functionality for volumes

{$ include 'pygfx.image_sample.wgsl' $}


struct VolGeometry {
    indices: array<i32,36>,
    positions: array<vec3<f32>,8>,
    texcoords: array<vec3<f32>,8>,
};

fn get_vol_geometry() -> VolGeometry {
    let size = u_wobject.volume_dimensions;
    var geo: VolGeometry;

    geo.indices = array<i32,36>(
        0, 1, 2, 3, 2, 1, 4, 5, 6, 7, 6, 5, 6, 7, 3, 2, 3, 7,
        1, 0, 4, 5, 4, 0, 5, 0, 7, 2, 7, 0, 1, 4, 3, 6, 3, 4,
    );

    let pos1 = vec3<f32>(-0.5);
    let pos2 = vec3<f32>(size) + pos1;
    geo.positions = array<vec3<f32>,8>(
        vec3<f32>(pos2.x, pos1.y, pos2.z),
        vec3<f32>(pos2.x, pos1.y, pos1.z),
        vec3<f32>(pos2.x, pos2.y, pos2.z),
        vec3<f32>(pos2.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos2.z),
        vec3<f32>(pos1.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos2.y, pos2.z),
    );

    geo.texcoords = array<vec3<f32>,8>(
        vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 1.0),
    );

    return geo;
}