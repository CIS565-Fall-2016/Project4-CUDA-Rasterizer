CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)


Without bounding box
Cow - 318 - 330 ms per frame

With bb
1.18 ms Max--largest triangles on screen -- per frame
Measured using cuda events

Side note: random speckles in image were caused by depth testing/lock issue.
Using a mutex array solved it.

#### Shading
Blinn-phong used from https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model


#### Blur
72-72 ms per frame - 21x21 - 8x8 blocks
60-61 ms per frame - 21x21 - 16x16 blocks
16x16 shared memory 

Without framebuffer[index] +=.... 14ms per frame
With shared memory...13ms per frame -- WRONG (bug in code)

420 global accesses, 21 shared.... (for one random pixel)
- 21x21 blur is bigger than blocks, so very little shared memory used

11x11 blur
- 4.23 - 4.25ms, 

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
