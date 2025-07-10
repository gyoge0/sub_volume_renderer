struct RenderOutput {
    color: vec4<f32>,
    coord: vec3<f32>,
};

// raycasting function for MIP rendering.
fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
    // Ideas for improvement:
    // * We could textureLoad() the 27 voxels surrounding the initial location
    //   and sample from that in the refinement step. Less texture loads and we
    //   could do linear interpolation also for formats like i16.
    // * Create helper textures at a lower resolution (e.g. min, max) so we can
    //   skip along the ray much faster. By taking smaller steps where needed,
    //   it will be both faster and more accurate.

    let nstepsf = f32(nsteps);

    // Primary loop. The purpose is to find the approximate location where
    // the maximum is.
    var the_ref = -999999.0;
    var the_coord = start_coord;
    var the_value : vec4<f32>;
    for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
        let coord = start_coord + iter * step_coord;
        let value = sample_vol(coord, sizef);
        let reff = value.r;
        if (reff > the_ref) {
            the_ref = reff;
            the_coord = coord;
            the_value = value;
        }
    }

    // Secondary loop to close in on a more accurate position using
    // a divide-by-two approach.
    var substep_coord = step_coord;
    for (var iter2=0; iter2<4; iter2=iter2+1) {
        substep_coord = substep_coord * 0.5;
        let coord1 = the_coord - substep_coord;
        let coord2 = the_coord + substep_coord;
        let value1 = sample_vol(coord1, sizef);
        let value2 = sample_vol(coord2, sizef);
        let ref1 = value1.r;
        let ref2 = value2.r;
        if (ref1 >= the_ref) {  // deliberate larger-equal
            the_ref = ref1;
            the_coord = coord1;
            the_value = value1;
        } else if (ref2 > the_ref) {
            the_ref = ref2;
            the_coord = coord2;
            the_value = value2;
        }
    }

    // Colormapping
    let color = sampled_value_to_color(the_value);
    // Move to physical colorspace (linear photon count) so we can do math
    $$ if colorspace == 'srgb'
        let physical_color = srgb2physical(color.rgb);
    $$ else
        let physical_color = color.rgb;
    $$ endif
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // Produce result
    var out: RenderOutput;
    out.color = out_color;
    out.coord = the_coord;
    return out;
}
