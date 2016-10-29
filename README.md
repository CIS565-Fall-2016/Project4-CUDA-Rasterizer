CUDA Rasterizer
===============
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xueyin Wan
* Tested on: Windows 10 x64, i7-6700K @ 4.00GHz 16GB, GTX 970 4096MB (Personal Desktop)
* Compiled with Visual Studio 2013 and CUDA 7.5

## Project Objective
Use CUDA to implement a simplified rasterized graphics pipeline, similar to the OpenGL pipeline. 

## My Project Features
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


## Project Summary


### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
