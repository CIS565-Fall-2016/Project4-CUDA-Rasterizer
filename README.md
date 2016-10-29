CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

### Summary
For this project we implemented a graphics rasterizer. Like path-tracing, rasterization is a method for converting a specification of points in 3d space into a 2d image. Whereas a path-tracer simulates the movement of light rays through space, a rasterizer uses matrix transforms to project 3d objectives onto the screen. Also, instead of representing objects as Platonic solids as in the path tracer, a rasterizer decomposes objects into smaller "primitives", usually triangles.

Our basic pipeline is as follows:
 Vertex Assembly -> Vertex Transform -> Primitive Assembly -> Rasterization -> Fragment Shading

It's worth noting that while most rasterizer pipelines look something like this, there are variations from one to the next.

## Vertex Assembly
The objects that a rasterizer uses require assembly. Initially, a GLTF loader iterates through various "meshes" or collections of associated points and fill buffers with

- positions (in model space) of vertices
- normals (also in model space)
- pointers to shared texture images
- coordinates into these shared texture images
- vertices (associated with triangles in the next step)
- indices for associating vertices with primitives

Vertex assembly uses common indices to fill all the values associated with each vertex struct. These include:
- positions
- normals
- texture coordinates

## Vertex Transform
Our rasterizer essentially combines this step with the previous, since we transform vertex positions while assigning them to vertex structs. Transformation is actually not a single step.

Positions start in world space, in which the origin is at an arbitrary global position.

Next they are transformed into view space, where the origin is at the camera. This space is primarily used during vertex shading, when positions relative to the viewer and relative to light sources are taken into account.

Next they are transformed into "clip" space, in which object are projected onto the 2d plane of the screen, but parts that extend past the edge of the screen have not yet been clipped. This space is sometimes also called NDC space for "Normalized Device Coordinate" space. Finally, we scale this space so that the origin is at the lower left corner of the screen and a unit corresponds to a pixel.

## Primitive Assembly
This step simple associates vertices with triangles (or whatever primitive is being used). The indices mentioned earlier map each vertex to its parent primitive.

## Rasterization
This step actually accounts for the bulk of the code, although the tasks it performs are seemingly trivial: rasterization takes on two challenges: coverage and occlusion.

# Coverage
Coverage is mapping the vertices of a primitive to pixels that fall within the area of the primitive. We use the AABB method, where we search within the smallest possible bounding box that surrounds a primitive. Specifically, we scan from the upper left of the box and stop at the lower right, testing each pixel to see if it falls within the primitive and assigning a fragment to it if it does.

# Occlusion
Points in space should only be rendered if other objects don't obstruct them. To check for occlusion we use "depth testing." We use a "Z Buffer" with size proportional to the number of pixels (we scale up in the case of supersampling). For each sample of the screen, we update the Z buffer with an integer that corresponds to the depth of the closest point with larger values corresponding to greater distances. Usually we measure these depths from 0 to INT_MAX. Once all the depths have been updated, we iterate back through the sampled points and only assign fragments where the depth of the fragment corresponds to the depth in the Z buffer -- this indicates that the fragment was closer than any other at that sample point.

# Fragment Shading
During this part, we shade the base color of pixels to reflect material properties or lighting. We used a standard technique: Blinn-Phong shading, in which the intensity of the lighting is proportional to the angle between the surface normal and the mean of light angle and view angle. Thus light is brightest when a viewer views an object head on with a light source at the camera. Surfaces are dark, essentially when an object is lit from behind.

## Additional Features
# Texture Mapping.
In order to apply more complex color patterns, our rasterizer gives each primitive a pointer to a texture image. A texture image looks a bit like a smashed version of the object. Each vertex is assigned a "texture coordinate" that points to the spot in the texture image where the object gets its base color. In order to assign colors to fragments which are usually between vertices (not exactly at them) we simply interpolate the texture coordinates of all three vertices using barycentric coordinates.

Barycentric coordinates associated with a triangle (as in our case) have values that are proportional to a points nearness to each vertex of the triangle. For example, if a point is colocated with the third vertex, its barycentric coordinate would be (0, 0, 1). Moreover, a point falls inside a triangle only if all three of its barycentric coordinates are in the range [0, 1].

Interpolation over barycentric coordinates simply involves weighting the contribution of each vertex by the value of the barycentric coordinate associated with it.

Here is an image of textures applied to a duck and to a milk truck:

IMAGES

The main performance cost of texture mapping is the requirement to repeatedly access global memory, both for texture coordinates and the texture image itself. However, an arbitrarily complex texture can be used with only minor additional cost in memory.

Since the texture coordinates of the three vertices are repeatedly accessed by the pixels that fall within them, this is a feature that would strongly benefit from the use of shared memory.

# Antialiasing
Unlike previous efforts, we this time used randomization to perform antialiasing. Antialiasing is a process wherein the value assigned to a pixel is actually the average of several colors calculated from points within the pixel. These points are called "samples" and the technique of taking multiple samples per pixel is known as "supersampling." The result may be seen below:

IMAGES

Like texture mapping, antialiasing comes with a performance cost -- probably an even more significant one, actually. In general runtime scales with the number of samples taken per pixel as demonstrated by this chart:

Two major memory optimizations include:
1. Only sampling at edges, this the effects of aliasing are really only observable there.
2. Directly averaging colors in place in the fragment buffer, instead of increasing the size of the fragment buffer, assigning separate samples to separate indices and subsequently averaging. This proved tricky, since multiple threads would have to access the same index in the fragment buffer simultaneously leading to race conditions.
