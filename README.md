CUDA Rasterizer
===============
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xueyin Wan
* Tested on: Windows 10 x64, i7-6700K @ 4.00GHz 16GB, GTX 970 4096MB (Personal Desktop)
* Compiled with Visual Studio 2013 and CUDA 7.5

## Project Objective
Use CUDA to implement a simplified rasterized graphics pipeline, similar to the OpenGL pipeline. 

## My CUDA Rasterizer Features
### About Graphics Pipeline
* Vertex shading
* Primitive assembly with support for triangles read from buffers of index and vertex data
* Rasterization
* Fragment shading
* A depth buffer for storing and depth testing fragments
* Fragment-to-depth-buffer writing (with atomics for race avoidance)

### Extra Coolness
* Lambert and Blinn-Phong shading
* UV texture mapping with bilinear texture filtering 
* UV texture mapping with perspective correct texture coordinates
* Support rasterizing Triangles, Lines, Points
* Super Sampling Anti-Aliasing (SSAA)

## Project Introduction
As we all know, rasterizeration converts vector graphics into dot matrix graphics. It is quite popular and important in real-time rendering area. Modern 3D rendering APIs like OpenGL, DirectX (Microsoft), Vulkan (Khronos, quite new area) are all implemented related to rasterize techniques. 

Different from ray tracing technique (as my last project shows), there's no concept of shooting rays during the whole procedure.
The whole graphics pipeline is like this:


* 1. Vertex Assembly

Pull together a vertex from one or more buffers


* 2. Vertex Shader

Transform incoming vertex position from model to clip coordinates

`World Space ->  View Space -> Clipping Space -> Normalized Device Coordinates (NDC) Space -> Viewport (Screen/Window) Space`

* 3. Primitive Assembly

A vertex shader processes one vertex.  Primitive asstriangleembly groups vertices forming one primitive, e.g., a, line, etc.

* 4. Rasterization

Determine what pixels a primitive overlaps

* 5. Fragment Shader

Shades the fragment by simulating the interaction of light and material.
Different rendering scheme could be applied here: Blinn-Phong, Lambert, Non-Photorealistic Rendering (NPR), etc.

* 6. Per-Fragment Tests

Choose one candidate to fill framebuffer!
Common techniques : Depth test, Scissor test, etc.

* 7. Blending

Combine fragment color with framebuffer color

* 8. Framebuffer
Write color to our framebuffer, a.k.a tell each pixel its color :)


### Credits

* [UPENN CIS-565 GPU Programming : course & recitation slides](https://github.com/CIS565-Fall-2016) by [@Patrick](https://github.com/pjcozzi) [@Shrek](https://github.com/shrekshao) [@Gary](https://github.com/likangning93) 
* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
