CUDA Rasterization Pipeline
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Daniel Krupka
* Tested on: Debian testing (stretch), Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz 8GB, GTX 850M


# About
This is a CUDA rasterization pipeline made for UPenn's CIS565. It features
* GLTF model loading
* Perspective correct diffuse PNG textures
* Screen space ambient occlusion (SSAO)
The pipeline utilizes deferred shading - the `depthPass` kernel performs depth testing and
assembles the G-buffer prior to shading and SSAO.

# Screenshots
Video of the rasterizer in action can be found [here](https://youtu.be/_Y-9eAgICrI).
![Normals](renders/duck-normals.png "Duck Normals")
![Lambert](renders/duck-lambert.png "Duck Lambert")


# Textures
I implemented perspective-correct UV texturing for diffuse coloring, using CUDA's texture memory functionality.
![DuckTex](renders/duck-texture.png "Duck Texture")
![Truck](renders/truck-nossao.png "Duck Texture")

# SSAO
I also implemented SSAO, both using shared memory and without.
![Truck SSAO only](renders/truck-ssao-only.png "Duck Texture")
![Truck SSAO](renders/truck-ssao.png "Duck Texture")

# Performance
![notex nosm](renders/plt_loc_notex.png "Duck Texture")
![tex nosm](renders/plt_loc_tex.png "Duck Texture")
![tex sm](renders/plt_sm_tex.png "Duck Texture")
Interestingly, the
