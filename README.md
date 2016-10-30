CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ruoyu Fan
* Tested on:
  * Windows 10 x64, i7-4720HQ @ 2.60GHz, 16GB Memory, GTX 970M 3072MB (personal laptop)
  * Visual Studio 2015 + CUDA 8.0

Tile-based rasterization: known issue is... artifacts when too many primitives in a tile

![duck_tile_based](/screenshots/duck_tile_based.gif)

With 2*2 SSAA | Without SSAA
:-------------------------:|:-------------------------:
![duck_no_ssaa](/screenshots/duck_ssaa.jpg)  |  ![duck_ssaa](/screenshots/duck_no_ssaa.jpg)

Thanks my girlfriend, who pointed out a bug in my depth test

### Things I have done

Basics:
* Vertex shading.
* Primitive assembly with support for triangles read from buffers of index and
  vertex data.
* Rasterization.
* Fragment shading.
* A depth buffer for storing and depth testing fragments.
* Fragment-to-depth-buffer writing.
* Lambert lighting scheme.

Some more:
* Tile-based rasterization
* SSAA

#### SSAA

I implemented supersample antialiasing, which is configurable by `SSAA_LEVEL` flag in `rasterize.cu`. My SSAA implementation is done by simply multiplying the width and height for render buffers with SSAA_LEVEL, and use average color to scale down at `sendImageToPBO`.

Below is comparison between `SSAA_LEVEL 2` and `SSAA_LEVEL 1` (no antialiasing):

With 2*2 SSAA | Without SSAA
:-------------------------:|:-------------------------:
![duck_no_ssaa](/screenshots/duck_ssaa.jpg)  |  ![duck_ssaa](/screenshots/duck_no_ssaa.jpg)

And performance comparison different SSAA levels, using `duck.gltf`:
![chart_ssaa](/images/chart_ssaa.png)

| SSAA level | milliseconds per frame |
|------------|------------------------|
| 1x1        | 1.2439                 |
| 2x2        | 5.10771                |
| 3x3        | 11.2783                |
| 4x4        | 19.9311                |

The conclusion is that render time per frame increase basically linearly with sample count. Since SSAA is done by changing the size of the actual rendered frame.

#### Tile-Based Rasterization

__Tile-based rasterization__ is configurable by `TILE_BASED_RASTERIZATION` flag in `rasterize.cu`.

Using one thread per primitive when doing rasterization can have some drawbacks like when there are big triangles occupying the screen, the whole program is waiting for it to be rasterized to fragments, while actually only one thread is working on it.

In my tile-based rasterization, I use a tile buffer to divide the frame into equally sized tiles (fixed size at the moment), and I add every primitive that overlap a tile into its buffer. Then, the kernel function for rasterization is launched by per-tile level.

Performance comparison between __tile based rasterization__ and __per-primitive rasterization__ :

![chart_tile](/images/chart_tile.png)

| Scene                                             | milliseconds per frame |
|---------------------------------------------------|------------------------|
| Cow, 2x2 SSAA, tile based                         | 31.4                   |
| Cow, 2x2 SSAA, per-permitive                      | 4.8                    |
| CheckerBoard, 2x2 SSAA, close view, tile based    | 12.5                   |
| CheckerBoard, 2x2 SSAA, close view, per-primitive | 738.6                  |

Tile based rasterization does a good job on big primitives, but when there are many small primitives, per-primitive rasterization can be faster.

One reason for tile based rasterization is not fast enough may be that I am copying primitives to tile buffers, and if multiple tiles are sharing a primitive, I was copying the primitive multiple times (while still accessing them in global memory during rasterization)... One improvement will be storing just indices for primitive buffer at tile buffer, another will be using shared memory for tiles

##### Known Issue:
Currently, primitives numbers for each tile is fixed at a maximum of 64 in my implementation, and it stops accepting new primitives full, which may lead to some artifacts if scene is complex.  

![engine_incorrect](/screenshots/engine_incorrect.gif)

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
