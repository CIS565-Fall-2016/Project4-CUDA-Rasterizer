CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Bowen Bao
* Tested on: Windows 10, i7-6700K @ 4.00GHz 32GB, GTX 1080 8192MB (Personal Computer)

## Overview

Here's the list of features of this project:

1. Core Features (Basic Rasterizer):
	* Vertex shading
	* Rasterization
	* Fragment shader
2. Extra Features:
	* UV texture mapping with bilinear texture filtering and perspective correct texture coordinates.
	* SSAA
	* MSAA

![](/img/overall.png)

## Texture Mapping
Base on depth per fragment for each pixel, and the transformed texture position indices, we load the correspond texture color.
###Bilinear filtering
We use bilinear filtering to smooth textures when they are displayed larger or smaller than they actually are.

###Perspective Correct
There will be distortions if we transform the pixel coordinate directly to position on textures. In perspective correct mapping we take into consideration the depth of the coordinates as well.

![](/img/truck_first.png)

Result of distortion mapping.

![](/img/truck_pers_bilinear.png)

Result of perspective correct mapping and bilinear filtering.

## Anti-aliasing
### SSAA
We first perform 4xSSAA to remove most of the rough edges. Although the performance cost is huge.

![](/img/truck_ssaa_comp.png)

No anti-aliasing compared to SSAA.

### MSAA
We also perform 4xMSAA to remove the rough edges. We could observe MSAA detects the edges of different fragments, as most of the rough edges occurs on those edges. Running super sampling on only those pixels greatly reduce the performance cost compared to SSAA.

![](/img/truck_msaa_3.png)

Pixels in red are those whose super sample positions lie in different fragments. 

![](/img/truck_msaa_comp.png)

No anti-aliasing compared to MSAA.

## Performance Analysis
Here we compare the performance of our rasterizer under basic feature, 4xSSAA and 4xMSAA.

![](/img/performance.png)

This is the performance graph of rendering the milk truck. In MSAA we separate and perform depth test before rasterization, in order to render pixels based on nearby positions later in rasterization step. We can observe that MSAA is much faster than SSAA, as we are only super sampling pixels on fragment edges.

![](/img/performance_duck.png)

This is the performance comparison of rendering duck. It shows similar results as the earlier graph. Also we could observe the proportion of processing rasterization is not as large as rendering milk truck. This is because in this case the fragment pixel count is not as large as the previous example.

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
