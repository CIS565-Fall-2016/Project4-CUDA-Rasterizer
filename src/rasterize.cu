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
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TEXTURE 1
#define BILINEAR 0
#define CULLING 1
#define NPR 1

#define RADIUS 4
#define INTENSITY 25

namespace Rasterizer {

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
#if TEXTURE == 1
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
#endif
		// ...
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
#if TEXTURE == 1
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int texWidth, texHeight;
#endif
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
#if TEXTURE == 1
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
#endif
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}
using namespace Rasterizer;

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static Fragment *dev_nprBuffer = NULL;

static float * dev_depth = NULL;
static int * dev_mutex = NULL;
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

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		auto & frag = fragmentBuffer[index];
		
		framebuffer[index] = frag.color * glm::max(0.0f, glm::dot(frag.eyeNor, glm::normalize(glm::vec3(1.0f)))) +
			0.1f * frag.color;
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));

	cudaFree(dev_nprBuffer);
	cudaMalloc(&dev_nprBuffer, width * height * sizeof(Fragment));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
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
#if TEXTURE == 1
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;

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

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}
#endif

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
#if TEXTURE == 1
						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,
#endif
						dev_vertexOut	//VertexOut
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



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {
	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		primitive.dev_verticesOut[vid].pos = MVP * glm::vec4((glm::vec3) primitive.dev_position[vid], 1);
		
		if (fabs(primitive.dev_verticesOut[vid].pos.w) > EPSILON) {
			primitive.dev_verticesOut[vid].pos /= primitive.dev_verticesOut[vid].pos.w;
		}
		primitive.dev_verticesOut[vid].pos.x = 0.5f * (float)width * (primitive.dev_verticesOut[vid].pos.x + 1.0f);
		primitive.dev_verticesOut[vid].pos.y = 0.5f * (float)height * (primitive.dev_verticesOut[vid].pos.y + 1.0f);

		auto eyePos = MV * glm::vec4(primitive.dev_position[vid], 1.0f);
		if (fabs(eyePos.w) > EPSILON) {
			primitive.dev_verticesOut[vid].eyePos = glm::vec3(eyePos / eyePos.w);
		}
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		

#if TEXTURE == 1
		primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];		
		primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
		primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
		primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
#endif

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

__global__ void initDepths(int width, int height, float* dev_depth) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		dev_depth[idx] = FLT_MAX;
	}
}

__global__ void _rasterize(
	int totalNumPrimitives, 
	int width, int height, 
	Primitive* dev_primitives,
	Fragment* dev_fragmentBuffer,
	float* dev_depth,
	int* dev_mutex
	) {
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid > totalNumPrimitives) {
		return;
	}
	
	Primitive & prim = dev_primitives[pid];

	const glm::vec3 tri[3] = {
		glm::vec3(prim.v[0].pos),
		glm::vec3(prim.v[1].pos),
		glm::vec3(prim.v[2].pos)
	};
	AABB aabb = getAABBForTriangle(tri);
	int maxx = glm::clamp(0, width, ((int) aabb.max.x) + 1),
		maxy = glm::clamp(0, height, ((int) aabb.max.y) + 1);
	int fid;
	for (int i = glm::clamp(0, width, (int)aabb.min.x); i < maxx; i++) {
		for (int j = glm::clamp(0, height, (int)aabb.min.y); j < maxy; j++) {
			fid = (height - j - 1) * width + (width - i - 1);
			Fragment & frag = dev_fragmentBuffer[fid];
			glm::vec3 barycentric = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
			if (isBarycentricCoordInBounds(barycentric)) {
				float z = glm::dot(barycentric, glm::vec3(
					prim.v[0].pos.z, 
					prim.v[1].pos.z, 
					prim.v[2].pos.z)
				);
				bool isSet;
				do {
					isSet = (atomicCAS(&dev_mutex[fid], 0, 1) == 0);
					if (isSet) {
						if (z < dev_depth[fid]) {
							dev_depth[fid] = z;
							frag.eyePos = glm::mat3(prim.v[0].eyePos, prim.v[1].eyePos, prim.v[2].eyePos) * barycentric;
							frag.eyeNor = glm::mat3(prim.v[0].eyeNor, prim.v[1].eyeNor, prim.v[2].eyeNor) * barycentric;
#if TEXTURE == 1
							if (!prim.v[0].dev_diffuseTex) { // nothing to see here!
								frag.color = glm::vec3(1.0f);
								frag.dev_diffuseTex = NULL;
							}
							else {
								// affine perspective correction
								const float perspectiveDepth = 1.0f / (barycentric.x / prim.v[0].eyePos.z + barycentric.y / prim.v[1].eyePos.z + barycentric.z / prim.v[2].eyePos.z);
								frag.texcoord0 = glm::vec2(
									barycentric.x * prim.v[0].texcoord0 / prim.v[0].eyePos.z +
									barycentric.y * prim.v[1].texcoord0 / prim.v[1].eyePos.z +
									barycentric.z * prim.v[2].texcoord0 / prim.v[2].eyePos.z) * perspectiveDepth;
								frag.texHeight = prim.v[0].texHeight;
								frag.texWidth = prim.v[0].texWidth;
								frag.dev_diffuseTex = prim.v[0].dev_diffuseTex;
							}
#else
							frag.color = glm::vec3(1.0f);
#endif
						}
					}
					if (isSet) {
						dev_mutex[fid] = 0;
					}
				} while (z < dev_depth[fid] && !isSet);
			}
		}
	}


}


__device__ __host__
glm::vec3 getPixelColor(int x, int y, int w, int h, TextureData * texture) {
	if (x < 0 || x >= w || y < 0 || y >= h) {
		return glm::vec3(0.0f);
	}
	int texIdx = x + y * w;
	return glm::vec3(texture[3 * texIdx], texture[3 * texIdx + 1], texture[3 * texIdx + 2]) / 255.0f;
}

#if TEXTURE == 1
__global__ void _loadTextures(int w, int h, Fragment * dev_fragmentBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x > w || y > h) {
		return;
	}
	int index = x + (y * w);
	if (!dev_fragmentBuffer[index].dev_diffuseTex) {
		return;
	}
	Fragment & frag = dev_fragmentBuffer[index];
#if BILINEAR == 1
	const float u = frag.texcoord0.x * frag.texWidth - 0.5f;
	const float v = frag.texcoord0.y * frag.texHeight - 0.5f;
	const int ux = glm::floor(u);
	const int vy = glm::floor(v);
	const float u_ratio = u - ux;
	const float v_ratio = v - vy;
	const float u_opposite = 1.0f - u_ratio;
	const float v_opposite = 1.0f - v_ratio;
	glm::vec3 i00 = getPixelColor(ux, vy, frag.texWidth, frag.texHeight, frag.dev_diffuseTex);
	glm::vec3 i01 = getPixelColor(ux, vy+1, frag.texWidth, frag.texHeight, frag.dev_diffuseTex);
	glm::vec3 i10 = getPixelColor(ux+1, vy, frag.texWidth, frag.texHeight, frag.dev_diffuseTex);
	glm::vec3 i11 = getPixelColor(ux+1, vy+1, frag.texWidth, frag.texHeight, frag.dev_diffuseTex);
	frag.color = v_ratio * (u_ratio * i11 + u_opposite * i01) + v_opposite * (u_ratio * i10 + u_opposite * i00);

#else
	int idx = (int)(frag.texcoord0.x * frag.texWidth) + (int)(frag.texcoord0.y * frag.texWidth) * frag.texHeight;
	frag.color = (1.0f / 255.0f) * glm::vec3(
		frag.dev_diffuseTex[3 * idx],
		frag.dev_diffuseTex[3 * idx + 1],
		frag.dev_diffuseTex[3 * idx + 2]);
#endif
}
#endif

#if CULLING == 1
struct backfacing {
	__host__ __device__ bool operator()(const Primitive & p) {
		//return glm::dot(p.v[0].eyeNor, -p.v[0].eyePos) < 0.0f; does not work perfectly!
		return glm::cross(p.v[1].eyePos - p.v[0].eyePos, p.v[2].eyePos - p.v[0].eyePos)[2] < 0;
	}
};
#endif

#if NPR == 1
__global__ void _npr(int w, int h, Fragment* dev_fragmentBuffer, Fragment* dev_nprBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < RADIUS || x >= w - RADIUS || y < RADIUS || y >= h - RADIUS) {
		return;
	}
	int index = x + (y * w);
	int intensityCount[256] = { 0 };
	int rBuf[256] = { 0 };
	int gBuf[256] = { 0 };
	int bBuf[256] = { 0 };

	// iterate in a RADIUS around x,y
	for (int i = -RADIUS; i <= RADIUS; i++) { // i == y, j == x
		for (int j = -RADIUS; j <= RADIUS; j++) {
			glm::ivec3 ayy = dev_fragmentBuffer[x + j + (y + i) * w].color * 255.0f;
			// find intensity
			int curIntensity = ((ayy.r + ayy.g + ayy.b) / 3.0) * INTENSITY / 255;
			if (curIntensity > 255) curIntensity = 255;
			intensityCount[curIntensity]++;
			rBuf[curIntensity] += ayy.r;
			gBuf[curIntensity] += ayy.g;
			bBuf[curIntensity] += ayy.b;

		}
	}
	int curMax = 0;
	int maxIndex = 0;
	for (int i = 0; i < 256; i++) {
		if (intensityCount[i] > curMax) {
			curMax = intensityCount[i];
			maxIndex = i;
		} 
	}
	dev_nprBuffer[index].color = glm::vec3(rBuf[maxIndex], gBuf[maxIndex], bBuf[maxIndex]) / (255.0f * curMax);
}
#endif

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
	cudaMemset(dev_mutex, false, width * height * sizeof(bool));
	initDepths << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// Backface Culling
	int numPrimitives = totalNumPrimitives;
#if CULLING == 1
	auto dev_thrust_primitives = thrust::device_pointer_cast(dev_primitives);
	numPrimitives = thrust::remove_if(dev_thrust_primitives, dev_thrust_primitives + numPrimitives, backfacing()) - dev_thrust_primitives;
#endif

	// TODO: rasterize
	dim3 numThreadsPerBlock(128);
	dim3 numBlocksForPrimitives((numPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	_rasterize << <numBlocksForPrimitives, numThreadsPerBlock >> >(numPrimitives, width, height, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex);
#if TEXTURE == 1
	_loadTextures << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer);
#endif
#if NPR == 1
	cudaMemcpy(dev_nprBuffer, dev_fragmentBuffer, width * height * sizeof(Fragment), cudaMemcpyDeviceToDevice);
	_npr << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_nprBuffer);
	std::swap(dev_nprBuffer, dev_fragmentBuffer);

#endif
    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
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
#if TEXTURE == 1
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);
#endif
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
