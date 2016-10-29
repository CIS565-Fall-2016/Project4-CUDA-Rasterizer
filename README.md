CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ruoyu Fan
* Tested on:
  * Windows 10 x64, i7-4720HQ @ 2.60GHz, 16GB Memory, GTX 970M 3072MB (personal laptop)
  * Visual Studio 2015 + CUDA 8.0

Tile Based Rasterization | Per-Primitive Rasterization
:-------------------------:|:-------------------------:
![duck_tile_based](/screenshots/duck_tile_based.png)  |  ![duck_prim_based](/screenshots/duck_tile_based.png)

Without SSAA  | With 2*2 SSAA
:-------------------------:|:-------------------------:
![duck_no_ssaa](/screenshots/duck_no_ssaa.png)  |  ![duck_ssaa](/screenshots/duck_ssaa.png)

Thanks my girlfriend, who pointed out a bug in my depth test

### Things I have done

* Vertex shading.
* Primitive assembly with support for triangles read from buffers of index and
  vertex data.
* Rasterization.
* Fragment shading.
* A depth buffer for storing and depth testing fragments.
* Fragment-to-depth-buffer writing.
* Lambert lighting scheme.

* Tile-based rasterization
* SSAA

#### SSAA

I implemented supersample antialiasing, which is configurable by `SSAA_LEVEL` flag in `rasterize.cu`. My SSAA implementation is done by simply multiplying the width and height for render buffers with SSAA_LEVEL, and use average color to scale down at `sendImageToPBO`.

Below is comparison between `SSAA_LEVEL 2` and `SSAA_LEVEL 1` (no antialiasing):

Without SSAA  | With 2*2 SSAA
:-------------------------:|:-------------------------:
![duck_no_ssaa](/screenshots/duck_no_ssaa.png)  |  ![duck_ssaa](/screenshots/duck_ssaa.png)

And performance comparison different SSAA levels:
![duck_no_ssaa](/images/chart_ssaa.png)

#### Tile-Based Rasterization

__Tile-based rasterization__ is configurable by `TILE_BASED_RASTERIZATION` flag in `rasterize.cu`.

Using one thread per primitive when doing rasterization can have some drawbacks like when there are big triangles occupying the screen, the whole program is waiting for it to be rasterized to fragments, while actually only one thread is working on it.

Performance comparison between __tile based rasterization__ and __per-primitive rasterization__ (SSAA level 2):

Tile Based Rasterization | Per-Primitive Rasterization
:-------------------------:|:-------------------------:
![duck_tile_based](/screenshots/duck_tile_based.png)  |  ![duck_prim_based](/screenshots/duck_tile_based.png)

In my tile-based rasterization, I use a tile buffer to divide the frame into equally sized tiles (fixed size at the moment), and I add every primitive that overlap a tile into its buffer. Then, the kernel function for rasterization is launched by per-tile level.

##### Known Issue:
Currently, primitives numbers for each tile is fixed at a maximum of 64 in my implementation, and it stops accepting new primitives full, which may lead to some artifacts if scene is complex.  

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
