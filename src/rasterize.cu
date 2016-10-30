/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

#define SSAA 1
#define MSAA 0
#define MSAA_COF 0.5f

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		// glm::vec3 col;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
		int component;
		// ...
	};

	struct Light {
		float emittance;
		glm::vec4 pos;
		glm::vec3 eyePos;
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		// VertexAttributeTexcoord texcoord0;
		// TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
		int component;
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static unsigned int*dev_mutex = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static float * dev_depth = NULL;	// you might need this buffer when doing depth test

#if MSAA == 1
static int *dev_fragIdx = NULL;
#endif

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__device__
glm::vec3 getColorByIndex(int index, glm::vec3 *image)
{
	glm::vec3 color;
	color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
	color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
	color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
	return color;
}

/**
* Kernel that writes the image to the OpenGL PBO SSAA directly.
*/
__global__
void sendSSAAImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w/2);

	if (x < w/2 && y < h/2) {
		glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);

		int originalIdx = 2 * x + 2 * y * w;
		color += getColorByIndex(originalIdx, image);
		color += getColorByIndex(originalIdx + 1, image);
		color += getColorByIndex(originalIdx + w, image);
		color += getColorByIndex(originalIdx + w + 1, image);
		color *= 0.25f;

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, Light* light) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		// TODO: add your fragment shader code here
		//Fragment *frag = &fragmentBuffer[index];
		//printf("light %f \n", glm::dot(frag->eyeNor, glm::normalize(light->eyePos - frag->eyePos)));
		//framebuffer[index] = fragmentBuffer[index].color;
		framebuffer[index] = light->emittance * fragmentBuffer[index].color
			* glm::dot(fragmentBuffer[index].eyeNor, glm::normalize(light->eyePos - fragmentBuffer[index].eyePos));
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;

#if SSAA == 1
	width *= 2;
	height *= 2;
	printf("Applying 2xSSAA, width %d height %d.\n", width, height);
#endif

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
#if MSAA == 1
	cudaMalloc(&dev_depth, 4 * width * height * sizeof(float));
#else
	cudaMalloc(&dev_depth, width * height * sizeof(float));
#endif
	checkCUDAError("rasterizeInit depth");

	cudaFree(dev_mutex);
#if MSAA == 1
	cudaMalloc(&dev_mutex, 4*width*height*sizeof(unsigned int));
	cudaMemset(dev_mutex, 0, 4*width*height*sizeof(unsigned int));
#else
	cudaMalloc(&dev_mutex, width*height*sizeof(unsigned int));
	cudaMemset(dev_mutex, 0, width*height*sizeof(unsigned int));
#endif
	checkCUDAError("rasterizeInit mutex");

#if MSAA == 1
	cudaFree(dev_fragIdx);
	cudaMalloc(&dev_fragIdx, width*height * 4 * sizeof(int));
	cudaMemset(dev_fragIdx, 0, width*height * 4 * sizeof(int));
#endif

	checkCUDAError("rasterizeInit frag");
}

__global__
void initDepth(int w, int h, float * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = FLT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						//printf("num of vertices %d \n");
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					int component = 0;

					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;
									component = image.component;

									checkCUDAError("Set Texture Image data");
									printf("Texture data pt: %d\n", dev_diffuseTex);
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut,	//VertexOut
						component
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__device__
float transFloat(float x, int len)
{
	float new_x = (x + 1.0f) * 0.5f * (float)len;
	return new_x;
}

__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space


		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		auto *vout = &primitive.dev_verticesOut[vid];

		vout->pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
		vout->pos /= vout->pos.w;
		vout->pos.x = transFloat(vout->pos.x, width);
		vout->pos.y = transFloat(vout->pos.y, height);

		glm::vec4 eyePos = MV * glm::vec4(primitive.dev_position[vid], 1.0f);
		vout->eyePos = glm::vec3(eyePos / eyePos.w);
		vout->eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);

		if (primitive.dev_diffuseTex != NULL)
		{
			//printf("Primitive tex not null");
			vout->dev_diffuseTex = primitive.dev_diffuseTex;
			vout->texcoord0 = primitive.dev_texcoord0[vid];
			vout->texWidth = primitive.diffuseTexWidth;
			vout->texHeight = primitive.diffuseTexHeight;
			vout->component = primitive.component;
		}
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}

__device__ glm::vec3 getColor(int texx, int texy, VertexOut *triangle)
{
	int component = triangle[0].component;

	int texIdx = texy * triangle[0].texWidth + texx;
	texIdx *= component;

	return glm::vec3(triangle[0].dev_diffuseTex[texIdx], 
		triangle[0].dev_diffuseTex[texIdx + 1], 
		triangle[0].dev_diffuseTex[texIdx + 2]) / 255.0f;
}


__device__ glm::vec3 getColorByXY(float x, float y, VertexOut *triangle, int width, int height)
{
	glm::vec3 color;

	glm::vec3 trianglePos[3] = {
		glm::vec3{ triangle[0].pos },
		glm::vec3{ triangle[1].pos },
		glm::vec3{ triangle[2].pos }
	};
	glm::vec3 barycentricCoord = calculateBarycentricCoordinate(trianglePos, glm::vec2(x, y));

	// update frag
	if (!isBarycentricCoordInBounds(barycentricCoord) || triangle[0].dev_diffuseTex == NULL || triangle[1].dev_diffuseTex == NULL || triangle[2].dev_diffuseTex == NULL)
	{
		//printf("no texture\n");
		// test texcoord;
		//glm::vec2 texcoord = glm::mat3x2(triangle[0].texcoord0, triangle[1].texcoord0, triangle[2].texcoord0) * barycentricCoord;
		//frag[idx].color = glm::vec3(texcoord.x, texcoord.y, 0);
		color = glm::vec3(0.0f, 0.0f, 0);
	}
	else
	{
		// texture
		//printf("texture\n");
		glm::vec3 persBarycentricCoord = glm::vec3(barycentricCoord.x / triangle[0].eyePos.z,
			barycentricCoord.y / triangle[1].eyePos.z, barycentricCoord.z / triangle[2].eyePos.z);
		glm::vec2 texcoord = glm::mat3x2(triangle[0].texcoord0, triangle[1].texcoord0, triangle[2].texcoord0) * persBarycentricCoord
			* (1.0f / glm::dot(glm::vec3(1.0f, 1.0f, 1.0f), persBarycentricCoord));
		// look at one point's texture.
		float texx_f = 0.5f + texcoord.x * (triangle[0].texWidth - 1);
		float texy_f = 0.5f + texcoord.y * (triangle[0].texHeight - 1);

		int texx = floor(texx_f);
		int texy = floor(texy_f);

		if (texx >= triangle[0].texWidth) texx = triangle[0].texWidth - 1;
		if (texy >= triangle[0].texHeight) texy = triangle[0].texHeight - 1;
		if (texx < 0) texx = 0;
		if (texy < 0) texy = 0;

		auto color00 = getColor(texx, texy, triangle);
		auto color10 = getColor(texx + 1, texy, triangle);
		auto color01 = getColor(texx, texy + 1, triangle);
		auto color11 = getColor(texx + 1, texy + 1, triangle);

		color00 = (texx_f - texx) * color10 + (1 - (texx_f - texx)) * color00;
		color01 = (texx_f - texx) * color11 + (1 - (texx_f - texx)) * color01;

		color = (texy_f - texy) * color01 + (1 - (texy_f - texy)) * color00;

		//color = glm::vec3(texcoord.x, texcoord.y, 0);
	}
	return color;
}

__global__ void _rasterization(int numIndices, Primitive *dev_primitive, int width, int height, Fragment *frag, unsigned int *fragMutex, float *fragDepth)
{
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices)
	{
		auto vec4ToVec2 = [](glm::vec4 p) -> glm::vec2 {
			return glm::vec2(p.x / p.w, p.y / p.w);
		};

		VertexOut *triangle = dev_primitive[iid].v;
		glm::vec3 trianglePts[] = { glm::vec3(triangle[0].pos), glm::vec3(triangle[1].pos), glm::vec3(triangle[2].pos) };
		auto aabbPts = getAABBForTriangle(trianglePts);

		auto trans = [](float x, int len) -> int
		{
			int new_x = x;
			if (new_x >= len) new_x = len - 1;
			if (new_x < 0) new_x = 0;
			return new_x;
		};

		int x_min = trans(aabbPts.min.x, width);
		int y_min = trans(aabbPts.min.y, height);
		int x_max = trans(aabbPts.max.x, width);
		int y_max = trans(aabbPts.max.y, height);

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				int idx = x + (height - y - 1) * width;

				glm::vec3 trianglePos[3] = {
					glm::vec3{ triangle[0].pos },
					glm::vec3{ triangle[1].pos },
					glm::vec3{ triangle[2].pos }
				};
				glm::vec3 barycentricCoord = calculateBarycentricCoordinate(trianglePos, glm::vec2(x, y));

				if (isBarycentricCoordInBounds(barycentricCoord))
				{
					bool isSet;
					do
					{
						isSet = (atomicCAS(&fragMutex[idx], 0, 1) == 0);
						if (isSet)
						{
							float depth = glm::dot(barycentricCoord, glm::vec3(triangle[0].pos.z, triangle[1].pos.z, triangle[2].pos.z));
							if (depth < fragDepth[idx])
							{
								// update frag
								if (triangle[0].dev_diffuseTex == NULL || triangle[1].dev_diffuseTex == NULL || triangle[2].dev_diffuseTex == NULL)
								{
									//printf("no texture\n");
									// test texcoord;
									//glm::vec2 texcoord = glm::mat3x2(triangle[0].texcoord0, triangle[1].texcoord0, triangle[2].texcoord0) * barycentricCoord;
									//frag[idx].color = glm::vec3(texcoord.x, texcoord.y, 0);
									frag[idx].color = glm::vec3(0.0f, 0.0f, 0);
								}
								else
								{
									// texture
									//printf("texture\n");
									glm::vec3 persBarycentricCoord = glm::vec3(barycentricCoord.x / triangle[0].eyePos.z, 
										barycentricCoord.y / triangle[1].eyePos.z, barycentricCoord.z / triangle[2].eyePos.z);
									glm::vec2 texcoord = glm::mat3x2(triangle[0].texcoord0, triangle[1].texcoord0, triangle[2].texcoord0) * persBarycentricCoord 
										* (1.0f / glm::dot(glm::vec3(1.0f, 1.0f, 1.0f), persBarycentricCoord));
									// look at one point's texture.
									float texx_f = 0.5f + texcoord.x * (triangle[0].texWidth - 1);
									float texy_f = 0.5f + texcoord.y * (triangle[0].texHeight - 1);

									int texx = floor(texx_f);
									int texy = floor(texy_f);

									auto color00 = getColor(texx, texy, triangle);
									auto color10 = getColor(texx + 1, texy, triangle);
									auto color01 = getColor(texx, texy + 1, triangle);
									auto color11 = getColor(texx + 1, texy + 1, triangle);

									color00 = (texx_f - texx) * color10 + (1 - (texx_f - texx)) * color00;
									color01 = (texx_f - texx) * color11 + (1 - (texx_f - texx)) * color01;

									auto color = (texy_f - texy) * color01 + (1 - (texy_f - texy)) * color00;
									
									// test coordinate
									//frag[idx].color = glm::vec3(texcoord.x, texcoord.y, 0);

									frag[idx].color = getColorByXY(x, y, triangle, width, height); //color;

								}
								frag[idx].eyePos = glm::mat3(triangle[0].eyePos, triangle[1].eyePos, triangle[2].eyePos) * barycentricCoord;
								frag[idx].eyeNor = glm::mat3(triangle[0].eyeNor, triangle[1].eyeNor, triangle[2].eyeNor) * barycentricCoord;
								fragDepth[idx] = depth;
							}
						}
						if (isSet)
						{
							fragMutex[idx] = 0;
						}
					} while (!isSet);
				}
			}
		}
	}
}


__device__ void _msaaUpdatePositionDepth(VertexOut *triangle, glm::vec3 *trianglePos, 
	float x, float y, float xdiff, float ydiff, int neiIdx, int iid, 
	Fragment *frag, unsigned int *fragMutex, float *fragDepth, int *fragIdx)
{
	glm::vec3 barycentricCoord = calculateBarycentricCoordinate(trianglePos, glm::vec2(x + xdiff, y + ydiff));
	if (isBarycentricCoordInBounds(barycentricCoord))
	{
		bool isSet;
		do
		{
			isSet = (atomicCAS(&fragMutex[neiIdx], 0, 1) == 0);
			if (isSet)
			{
				float depth = glm::dot(barycentricCoord, glm::vec3(triangle[0].pos.z, triangle[1].pos.z, triangle[2].pos.z));
				if (depth < fragDepth[neiIdx])
				{
					fragDepth[neiIdx] = depth;
					//printf("Setting fragment of index %d to %d\n", neiIdx, iid);
					fragIdx[neiIdx] = iid;
				}
			}
			if (isSet)
			{
				fragMutex[neiIdx] = 0;
			}
		} while (!isSet);
	}
}

__global__ void _msaaDepthTest(int numIndices, Primitive *dev_primitive, int width, int height, Fragment *frag, 
	unsigned int *fragMutex, float *fragDepth, int *fragIdx)
{
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices)
	{
		auto vec4ToVec2 = [](glm::vec4 p) -> glm::vec2 {
			return glm::vec2(p.x / p.w, p.y / p.w);
		};

		VertexOut *triangle = dev_primitive[iid].v;
		glm::vec3 trianglePts[] = { glm::vec3(triangle[0].pos), glm::vec3(triangle[1].pos), glm::vec3(triangle[2].pos) };
		auto aabbPts = getAABBForTriangle(trianglePts);

		auto trans = [](float x, int len) -> int
		{
			int new_x = x;
			if (new_x >= len) new_x = len - 1;
			if (new_x < 0) new_x = 0;
			return new_x;
		};

		int x_min = trans(aabbPts.min.x, width);
		int y_min = trans(aabbPts.min.y, height);
		int x_max = trans(aabbPts.max.x, width);
		int y_max = trans(aabbPts.max.y, height);

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				int neiIdx00 = x * 2 + (2 * height - 2 * y - 1) * width * 2;
				int neiIdx01 = neiIdx00 + 1;
				int neiIdx10 = neiIdx00 + 2 * width;
				int neiIdx11 = neiIdx10 + 1;

				glm::vec3 trianglePos[3] = {
					glm::vec3{ triangle[0].pos },
					glm::vec3{ triangle[1].pos },
					glm::vec3{ triangle[2].pos }
				};

				_msaaUpdatePositionDepth(triangle, trianglePos, x, y, 0,0 , neiIdx00, iid, frag, fragMutex, fragDepth, fragIdx);
				_msaaUpdatePositionDepth(triangle, trianglePos, x, y, MSAA_COF, 0, neiIdx01, iid, frag, fragMutex, fragDepth, fragIdx);
				_msaaUpdatePositionDepth(triangle, trianglePos, x, y, 0, MSAA_COF, neiIdx10, iid, frag, fragMutex, fragDepth, fragIdx);
				_msaaUpdatePositionDepth(triangle, trianglePos, x, y, MSAA_COF, MSAA_COF, neiIdx11, iid, frag, fragMutex, fragDepth, fragIdx);

			}
		}
	}
}

__global__ void _msaaRasterization(int numIndices, Primitive *dev_primitive, int width, int height, Fragment *frag, float *fragDepth, int *fragIdx, unsigned int *fragMutex)
{
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices)
	{
		auto vec4ToVec2 = [](glm::vec4 p) -> glm::vec2 {
			return glm::vec2(p.x / p.w, p.y / p.w);
		};

		VertexOut *triangle = dev_primitive[iid].v;
		glm::vec3 trianglePts[] = { glm::vec3(triangle[0].pos), glm::vec3(triangle[1].pos), glm::vec3(triangle[2].pos) };
		auto aabbPts = getAABBForTriangle(trianglePts);

		auto trans = [](float x, int len) -> int
		{
			int new_x = x;
			if (new_x >= len) new_x = len - 1;
			if (new_x < 0) new_x = 0;
			return new_x;
		};

		int x_min = trans(aabbPts.min.x, width);
		int y_min = trans(aabbPts.min.y, height);
		int x_max = trans(aabbPts.max.x, width);
		int y_max = trans(aabbPts.max.y, height);

		//printf("Debug:: Enter mass raster\n");

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				int idx = x + (height - y - 1) * width;

				//printf("DEBUG:: idx: %d\n", idx);

				glm::vec3 trianglePos[3] = {
					glm::vec3{ triangle[0].pos },
					glm::vec3{ triangle[1].pos },
					glm::vec3{ triangle[2].pos }
				};
				glm::vec3 barycentricCoord = calculateBarycentricCoordinate(trianglePos, glm::vec2(x, y));
				float depth = glm::dot(barycentricCoord, glm::vec3(triangle[0].pos.z, triangle[1].pos.z, triangle[2].pos.z));

				int neiIdx00 = x * 2 + (2 * height - 2 * y - 1) * width * 2;
				int neiIdx01 = neiIdx00 + 1;
				int neiIdx10 = neiIdx00 + 2 * width;
				int neiIdx11 = neiIdx10 + 1;

				//printf("DEBUG: fragidx: %d %d %d %d depth %f vs %f\n", fragIdx[neiIdx00], fragIdx[neiIdx01], fragIdx[neiIdx10], fragIdx[neiIdx11],
					//depth, fragDepth[neiIdx00]);

				if (isBarycentricCoordInBounds(barycentricCoord))
				{
					bool isSet;
					do
					{
						isSet = (atomicCAS(&fragMutex[idx], 0, 1) == 0);
						if (isSet)
						{
							float depth = glm::dot(barycentricCoord, glm::vec3(triangle[0].pos.z, triangle[1].pos.z, triangle[2].pos.z));
							if (depth < fragDepth[idx])
							{
								fragDepth[idx] = depth;
								//printf("Setting fragment of index %d to %d\n", neiIdx, iid);

								// this pixel belongs to this fragment.
								// check neighbors if they belong to different fragment.

								if (fragIdx[neiIdx00] == fragIdx[neiIdx01] &&
									fragIdx[neiIdx01] == fragIdx[neiIdx10] &&
									fragIdx[neiIdx10] == fragIdx[neiIdx11])
								{
									// proceed as usual.
									frag[idx].color = getColorByXY(x, y, triangle, width, height);
									//printf("DEBUG:: ind %d as usual %f %f %f\n", idx, frag[idx].color.r, frag[idx].color.g, frag[idx].color.b);
								}
								else
								{
									frag[idx].color = glm::vec3(1.0f, 0, 0);//getColorByXY(x, y, triangle, width, height);
									// for each position. check which triangle it belongs to. calculate color for that point.
									glm::vec3 neiCol00, neiCol01, neiCol10, neiCol11, color;

									neiCol00 = getColorByXY(x, y, dev_primitive[fragIdx[neiIdx00]].v, width, height);
									neiCol01 = getColorByXY(x + MSAA_COF, y, dev_primitive[fragIdx[neiIdx01]].v, width, height);
									neiCol10 = getColorByXY(x, y + MSAA_COF, dev_primitive[fragIdx[neiIdx10]].v, width, height);
									neiCol11 = getColorByXY(x + MSAA_COF, y + MSAA_COF, dev_primitive[fragIdx[neiIdx11]].v, width, height);

									color = getColorByXY(x, y, triangle, width, height);

									/*printf("Color of %d %d, %f %f %f, %f %f %f, %f %f %f, %f %f %f\n", x, y, neiCol00.r, neiCol00.g, neiCol00.b, neiCol01.r, neiCol01.g, neiCol01.b,
										neiCol10.r, neiCol10.g, neiCol10.b, neiCol11.r, neiCol11.g, neiCol11.b);*/

									frag[idx].color = (neiCol00 + neiCol01 + neiCol10 + neiCol11) * 0.25f;//0.125f + color * 0.5f;
								}
								frag[idx].eyePos = glm::mat3(triangle[0].eyePos, triangle[1].eyePos, triangle[2].eyePos) * barycentricCoord;
								frag[idx].eyeNor = glm::mat3(triangle[0].eyeNor, triangle[1].eyeNor, triangle[2].eyeNor) * barycentricCoord;

							}
						}
						if (isSet)
						{
							fragMutex[idx] = 0;
						}
					} while (!isSet);

				}
			}
		}
	}
}


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
#if MSAA == 1
	dim3 msaaBlockCount2d((width *2 - 1) / blockSize2d.x + 1,
		(height *2 - 1) / blockSize2d.y + 1);
	initDepth << <msaaBlockCount2d, blockSize2d >> >(width * 2, height * 2, dev_depth);
#else
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
#endif
	checkCUDAError("init depth");

	// TODO: rasterize
	//Primitive *primitives = new Primitive[totalNumPrimitives];
	//cudaMemcpy(primitives, dev_primitives, sizeof(Primitive) * totalNumPrimitives, cudaMemcpyDeviceToHost);
	dim3 numThreadsPerBlock(128);
	dim3 numBlocks((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
#if MSAA == 1
	cudaMemset(dev_mutex, 0, 4*width*height*sizeof(unsigned int));
	cudaMemset(dev_fragIdx, 0, 4*width*height*sizeof(int));
#else
	cudaMemset(dev_mutex, 0, width*height*sizeof(unsigned int));
#endif

#if MSAA == 1
	_msaaDepthTest << <numBlocks, numThreadsPerBlock >> >(totalNumPrimitives, dev_primitives, width, height,
		dev_fragmentBuffer, dev_mutex, dev_depth, dev_fragIdx);
	checkCUDAError("depth test");

	//printf("Depth test complete\n");
	initDepth << <msaaBlockCount2d, blockSize2d >> >(width * 2, height * 2, dev_depth);

	_msaaRasterization<<<numBlocks, numThreadsPerBlock>>>(totalNumPrimitives, dev_primitives, width, height,
		dev_fragmentBuffer, dev_depth, dev_fragIdx, dev_mutex);
#else
	_rasterization << <numBlocks, numThreadsPerBlock >> >(totalNumPrimitives, dev_primitives, width, height, 
		dev_fragmentBuffer, dev_mutex, dev_depth);
#endif
	checkCUDAError("rasterization");
	//printf("Finish one round of rasterization\n");

	// create temp light
	Light light;
	light.emittance = 2.0f;
	light.pos = glm::vec4(5.0f, 10.0f, 7.0f, 1.0f);
	glm::vec4 lightEyePos = MV * light.pos;
	light.eyePos = glm::vec3(lightEyePos / lightEyePos.w);
	Light *cudaLight;
	cudaMalloc(&cudaLight, sizeof(Light));
	cudaMemcpy(cudaLight, &light, sizeof(Light), cudaMemcpyHostToDevice);
	//printf("Finish one round of light\n");

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, cudaLight);
	checkCUDAError("fragment shader");
	//printf("Finish one round of render\n");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
#if SSAA == 1
	dim3 SSAABlockSize2d(sideLength2d, sideLength2d);
	dim3 SSAABlockCount2d((width / 2 - 1) / SSAABlockSize2d.x + 1,
		(height / 2 - 1) / SSAABlockSize2d.y + 1);

	sendSSAAImageToPBO << <SSAABlockCount2d, SSAABlockSize2d >> >(pbo, width, height, dev_framebuffer);
#else
	sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
#endif
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_mutex);
	dev_mutex = NULL;

    checkCUDAError("rasterize Free");
}
