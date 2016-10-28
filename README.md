CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Zimeng Yang
* Tested on: Windows 10, i7-4850 @ 2.3GHz 16GB, GT 750M (Personal Laptop)

### Features
* Rasterization
* Blinn-Phong shading
* Texture mapping with bilinear filter and perspective correct texture coordinates
* Rasterization for points and lines 

### Overview

![overview](renderings/overview.gif)

### Pipeline Overview

* Vertex Shading (apply Model, View, Projection transform and assembly vertexOut )
* Primitive Assembly (assembly vertices to primitives)
* Rasterization (fill in every fragment inside each primitive)
* Fragment Shading (apply fragment shading using pos, nor and texCoords information, BlinnPhong shading applied)
* Write Fragment Buffer to Frame Buffer (send current frame to display)

### Screenshots

#### Bilinear Texture Interpolation Comparison

Bilinear interpolation will reduce the aliasing of texture.

|without bilinear interp| with bilinear interp|
|----|----|
|![](renderings/checker_non_bilinear.PNG)|![](renderings/checker_bilinear.PNG)|

#### Perspective Correction

Applying perspective correction will make texture non-distorted.

|without perspective correction | with perspective correction|
|----|----|
|![](renderings/withoutPerspectiveCorrectness.PNG)|![](renderings/withPerspectiveCorrectness.PNG)|

#### Rasterization of Line and Point

Additional rasterization types:

for cow model:

|point|wireframe|solid|
|----|----|----|
|![](renderings/cow_point.PNG)|![](renderings/cow_wireframe.PNG)|![](renderings/cow_solid.PNG)|

for truck model:

|point|wireframe|solid|
|----|----|----|
|![](renderings/truck_point.PNG)|![](renderings/truck_wireframe.PNG)|![](renderings/truck_solid.PNG)|


### Performance Analysis

#### Different Models

![](PA/Models.png)

For these three different models, using solid rasterization. More complex the model is, more percentage the rasterization stage will take.
The percentage of render stage will decrease with the increasement of model complexity.

Details:

|duck.gltf|cow.gltf|truck.gltf|
|----|----|----|
|![](PA/duck_solid.PNG)|![](PA/cow_solid.PNG)|![](PA/truck_solid.PNG)|

#### Texture Mapping and Perspective Correct Coordinates

The performance impact of applying bilinear interpolation:

![](PA/BilinearInterpolation.png)

The performance impact of applying perspective correct coordinates:

![](PA/PerspectiveCorrect.png)

The bilinear interpolation will be applied during rendering stage. From the comparison above, the render stage percentage will increase due to the applying of bilinear interpolation.

The perspective correction will be applied during rasterization stage. Since it will introduce more computation and interpolations during correction, the rasterization state percentage increase drastically.

Details:

|no bilinear interp|no perspective correction|
|----|----|
|![](PA/truck_solid_non_bilinear.PNG)|![](PA/truck_solid_non_perspective_correct.PNG)|

#### Rasterization for Points and Lines

The three different rasterization type for same model: truck. (with bilinear texture interp and perspective correction)

![](PA/RenderMode.png)

Since number of fragments are increasing from point to line and then solid mode. The rasterization percentage will increase drastically too.

Details:

|points|lines|
|----|----|
|![](PA/truck_point.PNG)|![](PA/truck_wireframe.PNG)|

### Bloopers
hairy duck (line rasterization error):

![](renderings/hairyDuck.PNG)

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
