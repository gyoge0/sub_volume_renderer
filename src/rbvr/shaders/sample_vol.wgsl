fn sample_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = data_tex_coord * sizef;
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_ring_buffer));

    let n = u_wobject.ring_buffer_n * vec3<f32>(u_wobject.chunk_dimensions);
    let camera_pos = (u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let distance = data_coord - camera_pos;
    if (abs(distance.x) > n.x || abs(distance.y) > n.y || abs(distance.z) > n.z) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let wrapped_data_coord = data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_ring_buffer, vec3<i32>(wrapped_data_coord), 0);

    return result;
}

//
// fn sample_vol(
//     x: world coordinate
//     // this tells us how to wrap
//     offset: where it wraps
//     // defines where the texture is (this is the Roi)
//     b: start of the texture
//     size: size of texture
// ) -> vec4<f32>;
//
