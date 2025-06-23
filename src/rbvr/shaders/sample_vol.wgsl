fn sample_vol(tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = tex_coord * sizef;

    // casting to i32 truncates? or floors?
    let chunk = vec3<i32>(data_coord / vec3<f32>(u_wobject.chunk_dimensions));
    // this is passed as an f32, but it should be a u32. we can safely cast it to i32 to do math.
    let ring_buffer_dimensions = vec3<i32>(u_wobject.ring_buffer_dimensions_in_chunks);

    // this is
    // chunk (mod ring_buffer_dimensions)
    // where mod is a euclidian mod
    let ring_buffer_chunk = ((chunk % ring_buffer_dimensions) + ring_buffer_dimensions) % ring_buffer_dimensions;
    let chunk_offset = data_coord % u_wobject.chunk_dimensions;
    let sample_coord = vec3<f32>(ring_buffer_chunk) * u_wobject.chunk_dimensions + chunk_offset;

    let result = textureLoad(t_ring_buffer, vec3<i32>(sample_coord), 0);
//    let sample_tex_coord = sample_coord / vec3<f32>(textureDimensions(t_ring_buffer));
//    let result = textureSampleLevel(t_ring_buffer, s_ring_buffer, cache_tex_coord, 0.0);
    return result;
}