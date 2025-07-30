# Independent Renders

Currently, both rendering and loading are blocking operations on the
GPU. On each frame, we queue up all of our chunk loads and then we
perform a render operation. This results in long wait times between
renders when we are moving around. To improve the UX, we want to have
renders run seperately from texture updates.

## Multiple Queues

Most of the devices we're running on (Vulkan and Metal) support some
form of multiple queues which would process commands in a non-blocking
fashion. While this would solve our problem, WebGPU (and WGPU by
extension) doesn't support it yet ([multi-queue support has been
untouched for 3-5 years in 
WebGPU](https://github.com/gpuweb/gpuweb/issues?q=label%3Amulti-queue)).

## Priority Queues

Another alternative would be to implement a priority queue on top of the
[Device Queue](https://docs.rs/wgpu/latest/wgpu/struct.Queue.html).
However, device queues only support pushing commands and don't poll.
We'd have to speculatively submit commands to the queue, which is bit
too complex.

## Current Proposed Solution: Callback Based Submissions

WGPU exposes [a 
method](https://docs.rs/wgpu/latest/wgpu/struct.Queue.html#method.on_submitted_work_done)
to run a callback when the work currently submitted to a device queue
finishes. If we submitted individual texture load commands and waited
for the callback to trigger before sending more, we would avoid filling
up the queue with too many commands. If we tried to render while loading
textures, we would only have to wait for a single texture load command
to go through before rendering. 

```py
texture_load_queue = []

def load_next():
    device.load_texture(texture_load_queue.pop())
    device.on_submitted_work_done(load_next())
```

### Dealing with Pygfx

Pygfx provides the 
[update\_range](https://docs.pygfx.org/stable/_autosummary/resources/pygfx.resources.Texture.html#pygfx.resources.Texture.update_range)
method to signal updates to a texture. However, internally, pygfx delays
submitting update commands to the GPU until we render the next frame 
(see 
[\[1\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/resources/_texture.py#L476),
[\[2\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/resources/_base.py#L21-L22),
[\[3\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/renderer.py#L686-L687)).
This means that if we went through Pygfx, we'd still end up blocking for
texture updates on each render. To implement our solution, we'd need to
move our WrappingBuffer textures out of the Pygfx resource lifecycle and
manually send their update commands to the GPU.

# Race Condition in Loading New Data

Loading new data to the GPU is performed in chunks. If those chunks are
sent on a seperate thread, our render operation could happen in between
multiple chunk loads. This would mean the data available in the textures
is not all from the same location, creating artifacts on screen. Right
now, we don't run into this problem because we already block for texture
updates on the GPU.

## Swap Textures

One solution would be to use a second texture to load to, and then swap
it in after all our loads complete. This approach would take up double
the GPU memory used, since we'd need two different copies of our data on
the GPU. This would also require writing the entire wrapping texture on
each load. Right now, we only overwrite the sections in the texture that
have changed.

## Validity Flags

We could set some sort of flag for each chunk or pixel within our
wrapping texture indicating if it contains "good" data. The renderer
would then just read a 0 from those values instead of whatever is stored
in the texture. This could work, but it would also require extra writes
on each load.

## Other Ways to Minimize Artifacting

If we implement a priority for which chunks to load as well as a
fallback for missing data, we could minimize the artifacting seen on
screen. The priority would presumably be as follows:

```
low res near > low res far > high res near > high res far
```

By prioritising low resolutions, we load data that covers more physical
space.

# Weighted Average Rendering

One of Jan's wants for this renderer is to replace the MIP/LMIP with a
"weighted average" render mode, which would weight each sample by
distance. It would also allow better rendering across resolutions by
sampling from multiple resolutions near borders and fading them into
each other. Because the rays would be infinte (like in MIP and unlike
LMIP), we could even replace it with sampling a finite number of points
based on distance. We'd even know what texture to sample from beforehand
since we know the distance and the highest resolution that would be
loaded in.

# Swappable Rendering Pipeline

Ideally, we'd be able to swap out the raycasting and coloring
algorithims with different variants. For example, we might have a
dropdown in a future UI that lets users select from using MIP, LMIP, or
our "fading" algorithim. Pygfx currently does this via templating the
`raycast` function within the shader code. However, if our algorithims
are more complex, templating might not be the best solution. We might
need to introduce different materials that could indicate which
rendering mode we are in.

# Displaying Segmentations

Right now, we display segmentations by selecting a color based on the
segmentation ID of the point selected from our raycast algorithim.
However, this creates essentially iso boxes around cells.

## Tracing Objects

In case we wanted to show an outline around segmented objects, one way
to do it would be with multiple shader passes. We would have:

1. First fragment shader renders intensities to one texture and the
   labels to another texture
2. Use either an edge detection kernel or a builtin derivative function
   in WGSL to determine where the edges of the labels are
3. Use the results of the edge detection and the results of the
   intensity shader to produce a final image

This is relatively easy if we wrote raw WGPU code, but since we are
constrainted by the Pygfx render pipeline, it's a bit more difficult.
Pygfx provides a [`register_wgpu_render_function` 
decorator](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/utils.py#L16)
that can produce multiple shaders. These shaders appear to run in the
order we define them in (
[\[1\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/pipeline.py#L234-L265),
[\[2\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/renderer.py#L679-L700),
[\[3\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/renderer.py#L164-L173),
[\[4\]](https://github.com/pygfx/pygfx/blob/a9f65a26307373f3cf5dd305c2e00eda3bf6c25e/pygfx/renderers/wgpu/engine/renderer.py#L827-L850)
), so we could theoretically have multiple shader classes writing to and
reading from the same texture after each other. 

### Pygfx Post-Processing Effects

[Merged into main](https://github.com/pygfx/pygfx/pull/1114) (but not 
released yet), Pygfx has an API for creating post-processing effects,
which is essentially what we want to do with our multiple passes.
However, this is for the full screen and not just for our volume object.

## Without Selecting a Point

This discussion on displaying segmentations has been making the
assumption that we select a single point to display for each pixel on
screen and that we know the segmentation label of that point. However,
if we move to a different rendering method that doesn't select a single
point, this falls out of the window and we need to find a new way to
pick segmentations (or even just disallow them for that rendering
method).

## Default Coloring

Right now, many of the cells in the examples are colored red or have red 
pixels floating around. This is because unlabeled data has segmentation
id 0, which ends up selecting the first element within the color array
(which tends to be red). This should be fixed with some other default
color that is explicitly set as a default.

## Requiring Segmentations

Right now, all data requires segmentations passed in with it. The mouse
and multiscale examples both hijack this to instead show which scale
level we load from. If we intend to keep this, we should rename this
from "segmentations" to something generic like labels. Otherwise, we
should make this optional and write logic for operating without labels.

### `no-segmentations` Branch

The `no-segmentations` branch has a quick hack for turning off
segmentations. Actual implementations should use Jinja templating to
turn it off so that we can also choose to not bind the segmentation
textures. They should also reference the Pygfx volume renderer source
code for colormapping (or one of the earlier commits from before
segmentations being added).

# Pygfx Updates

Right now we pin Pygfx to `>=0.12,<0.13`. Pygfx is moving fast, so new
features are added constantly and as of July 2025 we're already 1
version behind. Since some of our proposed changes could end up
accessing Pygfx internals, it's important to update Pygfx carefully.

# Tests

The most logic heavy portion of the codebase (WrappingBuffer) has fairly
good test coverage. There are some other tests that try to work with
offscreen rendering, but those haven't been updated after some of the
breaking changes internally. Those should be fixed and we should add in
some more tests if we can.

## Hypothesis

Some of the tests use hypothesis for property testing. They've been
fairly useful in finding some bugs in the wrapping buffer
implementation. In general, if we have something logic heavy and
seperated from graphics heavy APIs, we should try to add in some
property tests.

# Time

Logically, time is just a 4th dimension on the WrappingBuffer. Even
though the current buffer only has 3 dimensions, adding a logical 4th is
trivial <!-- ;) -->. The problem is none of the graphics APIs support
having a 4d texture, so adding support in the shader is very not
trivial. 

## Multiple Textures

If we use a set of textures per timepoint (e.g., we store the previous 2
and the next 2) at the same time, we could hotswap the textures
(changing the bindings). However, this is a WGPU API that Pygfx doesn't
expose. It would also be very slow since we could only keep 5 timepoints
at a time.

### Sacrificing A Dimension

Because time is just another dimension, we could sacrifice movement in
one of the dimensions to buffer time in its place instead. This would
reduce our volume renderer back down to 2d, or we'd have to use this
other dimension as our sacrifical lamb to the GPU API gods and load only
4-5 points in it.
