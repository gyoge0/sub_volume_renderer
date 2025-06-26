fn sample_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = data_tex_coord * sizef;
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_ring_buffer));

    let tex_coord = data_coord / ring_buffer_dimensions;

    let result = textureSample(t_ring_buffer, s_ring_buffer, tex_coord);

    return result;
}