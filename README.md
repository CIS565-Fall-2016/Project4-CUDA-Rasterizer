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
* Correct color interpolation between points on a primitive

## Project Introduction
As we all know, rasterizeration converts vector graphics into dot matrix graphics. It is quite popular and important in real-time rendering area. Modern 3D rendering APIs like OpenGL, DirectX (Microsoft), Vulkan (Khronos, quite new area) are all implemented related to rasterize techniques. 

Different from ray tracing technique (as my last project shows), there's no concept of shooting rays during the whole procedure.
The whole graphics pipeline is like this: (Sort by stage order)

![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/readmepipeline.jpg "Graphics Pipeline") 

#### 1. Vertex Assembly

Pull together a vertex from one or more buffers

#### 2. Vertex Shader

Transform incoming vertex position from model to clip coordinates

`World Space ->  View Space -> Clipping Space -> Normalized Device Coordinates (NDC) Space -> Viewport (Screen/Window) Space`

#### 3. Primitive Assembly

A vertex shader processes one vertex.  Primitive asstriangleembly groups vertices forming one primitive, e.g., a, line, etc.

#### 4. Rasterization

Determine what pixels a primitive overlaps

#### 5. Fragment Shader

Shades the fragment by simulating the interaction of light and material.
Different rendering scheme could be applied here: Blinn-Phong, Lambert, Non-Photorealistic Rendering (NPR), etc.

#### 6. Per-Fragment Tests

Choose one candidate to fill framebuffer!
Common techniques : Depth test, Scissor test, etc.

#### 7. Blending

Combine fragment color with framebuffer color

#### 8. Framebuffer

Write color to framebuffer, a.k.a tell each pixel its color :)


## Showcase My Result
###Lambert and Blinn-Phong shading
|  Lambert  | Blinn-Phong shading |
|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/duck_with_lambert.gif "Lambert Duck") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/duck_with_blinn_phong_1.gif "Blinn Phong Duck") |

###UV texture mapping with bilinear texture filtering
We could see the apparant better result of Bilinear Texture Filtering since we take into account more texture infomation.

|  Without Bilinear Texture Filtering  | With Bilinear Texture Filtering |
|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/BilinearEnlarged.png "WithBininear") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/NoBilinearEnlarged.png "NoBininear") |

###UV texture mapping with perspective correct texture coordinates

We figure out the texture coordinates in a more sophisticated way. We interpolate texture coordinates, and then do the perspective divide.

|  Without Perspective Correctness | With Perspective Correctness |
|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/WithoutPerspectiveCorrectnessCheckerBoard.PNG "WithoutPerspective") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/PerspectiveCorrectnessCheckerBoard.PNG "WithPerspective") |

###Point Representation & Line Representation

|  Point Representation | Line Representation |
|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/truckpoint.gif "Truck Point") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/LineRepresentation.PNG "Truck Line") |

|  Point Representation | Line Representation |
|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/cowpoint.gif "Cow Point") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/cowcowcow.gif "Cow Line") |

###Super Sampling Anti-Aliasing (SSAA)
Divide one pixel into small pixels, then average to get the final color, put it into framebuffer

|  No SSAA | SSAA = 2 | SSAA = 4 |
|------|------|------|
|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/SSAAnoSHOW.PNG "No SSAA") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/SSAA2SHOW.PNG "SSAA 2 * 2") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/SSAA4SHOW.PNG "SSAA 4 * 4") |

###Correct color interpolation between points on a primitive

I use barycentric calculation of each triangle's vertex coordinate v[0], v[1] & v[2]'s color to fill each fragment color.

I use each vertex's normal in view(eye) coordinate system to represent its color in order to visualize.

|![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/normalInterpolate1.gif "Normal Interpolation") 

The two below are I accidently use the first triangle's vertex normal and create an interesting result. Also put here for fun.

| First Triangle Normal Interpolation | First Triangle Normal Interpolation | 
|------|------|
| ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/normalinterpolation.gif "First Triangle Normal Interpolation") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/cownormalinterpolation.gif "First Triangle Normal Interpolation") |


## Performance Analysis

###Different Stages of rasterizer time arrangement percentage(Compare between Lines, Points and Triangles)

From the picture below, we can see that when we choose triangle as our primitive, the rasterization stage occupies most of time since we do all the fragmentbuffer calculations & filling there, easpecially for the Axis-Aligned Bounding Box(AABB) calculation. In line & point cases, we need to do much less calculations w.s.t fragmentbuffer in rasterization stage, render case as a result plays a larger role to fill the framebuffer. 
![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/StagePerformance.PNG "Stage Perfomance")


###Compare between Super-Sampling Anti-Aliasing(SSAA) and No SSAA

Based on CG knowledge, we all know that SSAA is a time-consuming stage. Since we need to open a larger fragment buffer, frame buffer and depth buffer,  so the calculations maybe mutiple in order to get an average value then send it to PBO.
Here the result is based on CesiumMilkTruck.gltf, and my table is based on the time/fps.

![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/SSAAPerformance.PNG "SSAA Comparasion")


###Compare between Bilinear, Bilinear Texture Filtering,  Perspective Correctness Texture Filtering

Bilinear adds the percentage of the render stage during the whole cuda launch since we need to consider more pixels around to improve correctness(super sampling comes into play again)!
Perspective Correctness adds the percentage of rasterization, since it contains more interpolation and calculation during the rasterization procedure.

|   | Bilinear Open | Perspective Open | Nothing Opens|
|------|------|------|------|
|Rasterization Stage Percentage of GPU(%)| 55.69 | 59.36 | 53.98 |
|Render Stage Percentage of GPU(%)| 2.06 | 1.5 | 1.76 |


| Bilinear Open | Perspective Open | 
|------|------|
| ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/BilinearShow.PNG "Bilinear Open") | ![alt text](https://github.com/xueyinw/Project4-CUDA-Rasterizer/blob/master/results/PersShow.PNG "Perspective Open") |

### Credits

* [UPENN CIS-565 GPU Programming : course & recitation slides](https://github.com/CIS565-Fall-2016) by [@Patrick](https://github.com/pjcozzi) [@Shrek](https://github.com/shrekshao) [@Gary](https://github.com/likangning93) 
* UPENN CIS-561 Advanced Computer Graphics course slides
* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
