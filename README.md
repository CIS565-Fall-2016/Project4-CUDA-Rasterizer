CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Liang Peng
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz 2.6GHz 8GB, GTX 960M (Personal Laptop)

## Features
* [ ] Primitive
	* [ ] Point
	* [x] Line
	* [x] Triangle
* [ ] Texture
	* [x] Diffuse
	* [ ] Normal Map
* [x] Lighting
	* [x] Lambert
	* [x] Blinn-Phong
* [x] Normal Visualization
* [x] Depth Visualization
* [x] Depth Test with Mutex
* [x] Perspective Corrected Texcoord

## Overview
### Basic Attributes
Texture | Normal | Depth | Texcoord
--- | --- | --- | ---
![](img/cover_diffuse.gif) | ![](img/cover_normal.gif) | ![](img/cover_depth.gif) | ![](img/cover_texcoord.gif)

### Lighting
Lambert | Blinn-Phong
--- | ---
 ![](img/cover_lambert.gif) | ![](img/cover_blinnphong.gif)

### Rasterization Mode
Point | Wireframe | Solid
--- | --- | ---
 ![](img/point.gif) | ![](img/line.gif) | ![](img/triangle.gif)

### Depth Test with Mutex
Mutex OFF | Mutex ON
--- | ---
 ![](img/mutex_off.gif) | ![](img/mutex_on.gif)

### Texcoord Correction
Perspective Correction OFF | Perspective Correction ON
--- | ---
 ![](img/texcoord0.gif) | ![](img/texcoord1.gif)

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
