CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xiaomao Ding
* Tested on: Windows 8.1, i7-4700MQ @ 2.40GHz 8.00GB, GT 750M 2047MB (Personal Computer)

### Introduction
This code provides a rudimentary implentation of a rasterizer pipeline. A rasterizer takes information from a description of a scene (primitives, normals, colors) and renders it onto the computer screen. The code in this repository implements a basic pipeline, starting with a vertex shader -> primitive assembly -> rasterization -> fragment shading. The rasterization step also checks for the depth of the fragment so that only the nearest fragment is shaded. Unfortunately, I didn't have too much time to work on this project this, so code here is pretty bare-bones.

![](https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/Project4.gif)

Backface culling is enabled by setting the `CULLING` define at the top of `rasterize.cu` to 1.

### Base performance

I first tested the runtime for the rasterizer using the cow by varying the block size. The biggest difference in runtime came in the rasterization step where both 64 and 256 threads/block was much worse than 128 threads/block. I suspect this has something to do with block occupancy as the rasterization step is the only place where noticeable branching could occur. This might hold up the blocks when too many threads are allocated. For the 64 case, perhaps not allocating enough threads causes loss of performance due to stalling as there are not enough executable threads per warp.

<p align="center">
  <img src="https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/BlockSize.png?raw=true">
</p>

We can also look at the runtime for each of the four gltfs in this repo. Surprisingly, the box takes an incredible amount of time to rasterize! I am not sure what is going on here.

<p align="center">
  <img src="https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/NoCulling.png?raw=true">
</p>

### Backface culling

Backface culling is a method to determine whether a triangle is visible. We can use the order of the triangles vertices to determine whether it is facing toward or away from the camera. Triangles facing away are then "culled". This assumes that there are only closed and opaque objects in the scene as if this were not the case, parts of the scene may be missed during rendering. The plot below shows the runtimes of each object with the addition of the culling step. Unfortunately, it seems that the stream compaction (using `thrust::remove_if`) takes more time than that saved. However, in the case of the cow, the runtime is about on par to that without culling. This suggests that culling may only save runtime for scenes with many primitives.  For this project, it does fix some issues with strange rendering bugs that occur when not culling, described in the bloopers section below.

<p align="center">
  <img src="https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/Culling.png?raw=true">
</p>

### Bloopers
One bug that I haven't had the chance to sort out yet is the transparency issue. It appears that my use of atomicMin to clear race conditions in the rasterizer does not work perfectly, as if often renders fragments behind the front-most one! Additionally, for some reason, without backface culling, the rasterizer ends up rendering the fragments rather randomly, leading to jagged images.

Headless Duck    |  Odd Cow
:-------------------------:|:-------------------------:
![](https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/HeadlessDuck.PNG?raw=true)  |  ![](https://github.com/xnieamo/Project4-CUDA-Rasterizer/blob/master/images/wtfCow.PNG?raw=true)

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
