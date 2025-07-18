{$ include 'sub_volume.sample_vol.wgsl' $}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Get size of the volume
    let sizef = vec3<f32>(u_wobject.volume_dimensions);

    // Determine the stepsize as a float in pixels.
    // This value should be between ~ 0.1 and 1. Smaller values yield better
    // results at the cost of performance. With larger values you may miss
    // small structures (and corners of larger structures) because the step
    // may skip over them.
    // We could make this a user-facing property. But for now we scale between
    // 0.1 and 0.8 based on the (sqrt of the) volume size.
    let relative_step_size = clamp(sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0, 0.1, 0.8);

    // Positions in data coordinates
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    // Calculate unit vector pointing in the view direction through this fragment.
    let view_ray = normalize(far_pos - near_pos);

    // Calculate the (signed) distance, from back_pos to the first voxel
    // that must be sampled, expressed in data coords (voxels).
    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

    // Now we have the starting position. This is typically on a front face,
    // but it can also be incide the volume (on the near plane).
    let front_pos = back_pos + view_ray * dist;

    // Decide how many steps to take. If we'd not cul the front faces,
    // that would still happen here because nsteps would be negative.
    let nsteps = i32(-dist / relative_step_size + 0.5);
    if( nsteps < 1 ) { discard; }

    // Get starting position and step vector in texture coordinates.
    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord = ((back_pos - front_pos) / sizef) / f32(nsteps);

    // Render
    let render_out = raycast(sizef, nsteps, start_coord, step_coord);

    // Create fragment output.
    var out: FragmentOutput;
    out.color = render_out.color;
    return out;
}
