fn sample_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = data_tex_coord * sizef;
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_ring_buffer));


    let wrapped_data_coord = data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_ring_buffer, vec3<i32>(wrapped_data_coord), 0);

    return result;
}