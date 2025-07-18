# Background Knowledge

Throughout the process of creating this volume renderer, I've uncovered
a lot of useful resources and concepts. This document lists several of
them.

# Sparse Virtual Textures

Sparse Virtual Textures are a technique for creating massive logical
textures that are not loaded entirely onto the GPU. Different GPU APIs
provide ways to create them
([Vulkan](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_sparse_texture.txt),
[Metal](https://developer.apple.com/documentation/metal/creating-sparse-heaps-and-sparse-textures)).
This would allow the GPU to manage all the aspects of loading our large
textures, essentially taking the problem away from us. However, [WebGPU
doesn't support this yet because there hasn't been much interest on
it](https://github.com/gpuweb/gpuweb/issues/455).

## Custom Implementation

There are a couple guides online for implementing sparse virtual
textures by hand (see
[\[1\]](https://studiopixl.com/2022-04-27/sparse-virtual-textures),
[\[2\]](https://wickedengine.net/2024/06/texture-streaming/)). These
focus on 2D textures but the principles would still be the same for 3D.
The one issue (at least with \[1\]) is the global map of which chunks
have been loaded can become fairly large on its own. An alternative
might be to keep a map that's relative to the camera, which would mean
rewriting a smaller map every time we cross a chunk boundary.

## SVT + Mipmap

Mipmaps in textures precompute lower resolution versions of the textures
that can be sampled if we do not require the highest accuracy in
sampling (e.g., if the object is very far away and small on our screen).
Some of the APIs allow sparse textures to also be sparse within the
mipmaps, which would mean dynamically loading the best resolution
possible for any point sampled. *This is the goal of our WrappingBuffers
and custom `sample_vol` function!* If WebGPU and Pygfx implemented this,
most of the infrastructure provided by this project wouldn't be needed
since we could just have a single texture with a handle to our data.

# Distant Horizons

[Distant
Horizons](https://gitlab.com/distant-horizons-team/distant-horizons) is
a Minecraft mod that implements level of detail within Minecraft. This
combined with how Minecraft already chunks its world makes it a great
analogy of what we want to achieve, just with biological volume data.

# Coordinate Systems

All Pygfx objects (wobjects) provide a `.world` and `.local` property,
which are Affine Transforms. See
[here](https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html)
for a quick guide. In general, we tend to look at the `world` transforms
of different objects. Common transforms are exposed to the shader like
the camera and projection transform. These can be very useful when
converting between coordinate systems, but pay attention as to what
systems we are converting between before you use them.

In this renderer, we tend to work in a couple different coordinate
spaces:

- World space: the Pygfx world.
- Pixel space: within our data. 1 unit corresponds to 1 index within our
  backing data array. All pixels are assumed to be unit cubes; for data
  with pixels that aren't cubes, we simply scale the volume within the
  world. These might also be called texels (to avoid confusion with
  scree npixels)
- Chunk space: within our data and in chunks. 1 unit corresponds to 1
  chunk of data within our backing data array. We again assume each
  chunk to be a unit cube. We account for their true scale when
  converting into pixels and usually provide a uniform property with the
  size of a single chunk.

It is very important to keep track of what coordinate system we are
working in. Most variables in the WrappingBuffer are tagged with their
coordinate system somewhere in their name.

# C/Fortran Style

The shader code in WGSL all operates in Fortran style, where the fastest
changing index is the first index (e.g., the next element after `x[1, 5,
9]` would be `x[2, 5, 9]`. This is the *opposite* of what we are used to
in C-style languages (like Python) and NumPy: 

```py
x: gfx.Texture = ...
x.data[1, 5, 9] = 2.0
```
```wgsl
textureLoad(t_x, vec3<i32>(9, 5, 1)); // 2.0
```

This is problematic when we use data sent in buffers for calculating
indicies:

```py
x: gfx.Texture = ...
x.data[1, 5, 9] = 2.0
x.data[9, 5, 1] = 1.0

y: gfx.Buffer = ...
y.data["loc"] = np.array([1, 5, 9])

```
```wgsl
let y = u_y.loc; // vec3<i32>(1, 5, 9)
textureLoad(t_x, y); // 1.0!
```

The convention adopted for this project is that everything in Python
should be C-style and everything in WGSL should be Fortran style (i.e.,
no doing `.zyx` swizzles anywhere!). When we write data to textures, we
write using the numpy-like API (so we should just do `x.data[a, b, c] =
z[a, b, c]`). However, when we information used for calculating indices
to uniform buffers, we need to reverse them:

```py
# using the same x/y example from before:
y.data["loc"] = np.array([1, 5, 9])[::-1]
```

```wgsl
let y = u_y.loc; // vec3<i32>(9, 5, 1)
textureLoad(t_x, y); // 2.0, as expected
```

This standard matches what Pygfx seems to be doing internally (even if
it's not explicitly stated anywhere); accessing the `position` property
on any of the world transforms Pygfx provides seems to match up with
this standard.

> [!IMPORTANT]
> This fact is mostly determined experimentally by running examples with
> different sized volumes and seeing which dimension in
> `camera.world.position` changes the most. Always double check what
> coordinate system (`xyz`/C-style or `zyx`/Fortran-style) you are in,
> especially if you are getting weird bugs/out of bounds.


# `textureLoad` vs `textureSample`

Because adjacent texels in our textures might not be logically adjacent
(e.g., crossing chunk boundaries), we've forgone using a sampler for
interpolation right now. Instead, we use directly load texture values.
If we move to compacting our data into a single texture (with MIP
levels), we could look into using sampling again.
