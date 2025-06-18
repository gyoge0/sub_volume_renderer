fn sample_vol(tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = tex_coord * sizef;
    let chunk = data_coord / vec3<f32>(u_wobject.chunk_dimensions);
    let chunk_offset = data_coord % u_wobject.chunk_dimensions;
    let cache_chunk_coord = vec4<f32>(textureLoad(t_indirection, vec3<i32>(chunk), 0));
    if cache_chunk_coord.a != 1.0 {
        // The chunk is not loaded, return a transparent pixel
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let cache_coord = cache_chunk_coord.xyz + chunk_offset;
    let cache_tex_coord = cache_coord / vec3<f32>(textureDimensions(t_cache, 0));
    let result = textureSampleLevel(t_cache, s_cache, cache_tex_coord, 0.0);
    return result;
}