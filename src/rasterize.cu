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
#include <thrust/device_vector.h>

#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include <util/timer.h>

#include "rasterizeTools.h"
#include "rasterize.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

	typedef unsigned char BufferByte;

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;
	typedef glm::vec3 Color;

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
		
		 VertexAttributeTexcoord texcoord0;
	};

	struct Primitive
	{
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		
		VertexOut v[3];

		int texWidth, texHeight;
		TextureData* dev_diffuseTex = NULL;
	};

	struct Fragment {
		Color color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		
		int texWidth, texHeight;
		TextureData* dev_diffuseTex;
		VertexAttributeTexcoord texcoord0;
		float sobelXY;
	};

	struct PrimitiveDevBufPointers
	{
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
		int texWidth, texHeight;
		TextureData* dev_diffuseTex;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};
}

static int SOBEL_GRID_SIZE = 3;

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Primitive *dev_primitivesCulled = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static Color *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h)
{
    width = w;
    height = h;

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(Color));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(Color));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

/**
* Called once at the end of the program to free CUDA memory.
*/
void rasterizeFree() 
{
	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

	cudaFree(dev_framebuffer);
	dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	checkCUDAError("rasterize Free");
}

__global__
void initFrameBuffers(int w, int h, int * depth, Fragment *fragmentBuffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
		fragmentBuffer[index].color = Color(0.33, 0.33, 0.33);
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
	if (vid < numVertices) 
	{
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		
		if (normal)
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

void setSceneBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;
	checkCUDAError("Set BufferView Device Mem");
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
			checkCUDAError("Set BufferView Device Mem");
			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			checkCUDAError("Set BufferView Device Mem");
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

		int primitiveDrawMode = 0;
		switch (scene.drawModelType)
		{
		case tinygltf::Scene::Point:
			primitiveDrawMode = TINYGLTF_MODE_POINTS;
			break;
		case tinygltf::Scene::Line:
			primitiveDrawMode = TINYGLTF_MODE_LINE;
			break;
		case tinygltf::Scene::Triangle:
		default:
			primitiveDrawMode = TINYGLTF_MODE_TRIANGLES;
			break;
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
					
					(const_cast<tinygltf::Primitive*> (&primitive))->mode = primitiveDrawMode;

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
					checkCUDAError("Set Index Buffer");
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
						numPrimitives = 3 * (numIndices / 3);
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

						if (dev_attribute)
						{
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
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					int texWidth, texHeight;
					TextureData* dev_diffuseTex = NULL;
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
									
									texWidth = image.width;
									texHeight = image.height;

									checkCUDAError("Set Texture Image data");
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

						texWidth,
						texHeight,
						dev_diffuseTex,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;
				} // for each primitive

			} // for each mesh

		} // for each node

	}
	
	checkCUDAError("Free BufferView Device Mem");
	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{
		checkCUDAError("Free BufferView Device Mem");
		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}
}

void clearSceneBuffers()
{
	// deconstruct primitives attribute/indices device buffer
	checkCUDAError("Set BufferView Device Mem");
	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it)
	{
		for (auto p = it->second.begin(); p != it->second.end(); ++p)
		{
			cudaFree(p->dev_indices);
			checkCUDAError("Set BufferView Device Mem");
			cudaFree(p->dev_position);
			checkCUDAError("Set BufferView Device Mem");
			cudaFree(p->dev_normal);
			checkCUDAError("Set BufferView Device Mem");
			cudaFree(p->dev_texcoord0);
			checkCUDAError("Set BufferView Device Mem");
			cudaFree(p->dev_diffuseTex);
			checkCUDAError("Set BufferView Device Mem");
			cudaFree(p->dev_verticesOut);
			//TODO: release other attributes and materials
		}
	}
	mesh2PrimitivesMap.clear();

	checkCUDAError("Set BufferView Device Mem");
	cudaFree(dev_primitives);
	dev_primitives = NULL;
	checkCUDAError("Set BufferView Device Mem");
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
		
		// vtxPos: Where the original positions buffer read from model are.
		glm::vec4 vtxPos = glm::vec4(primitive.dev_position[vid], 1.0f);
		
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		glm::vec4 projVtxPos = MVP * vtxPos;
		// Then divide the pos by its w element to transform into NDC space. 
		const float projW = projVtxPos.w;
		projVtxPos /= projVtxPos.w; // All elements are now between (-1, -1, -1) ~ (1, 1, 1).
		// Finally transform x and y to viewport space
		glm::vec3 pixelPos(
		(float)width	* 0.5f * (-projVtxPos.x + 1.0f),
		(float)height	* 0.5f * (-projVtxPos.y + 1.0f),
						  0.5f * (projVtxPos.z + 1.0f));

		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].pos = glm::vec4(pixelPos, projW);
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * vtxPos);
		
		if (primitive.dev_normal)
			primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
		
		if (primitive.dev_texcoord0)
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		int pid = iid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES)
		{
			pid = iid / (int)primitive.primitiveType;

			dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			
			dev_primitives[pid + curPrimitiveBeginId].texWidth = primitive.texWidth;
			dev_primitives[pid + curPrimitiveBeginId].texHeight = primitive.texHeight;
			dev_primitives[pid + curPrimitiveBeginId].dev_diffuseTex = primitive.dev_diffuseTex;
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_LINE)
		{
			const int ptIdx = iid % 3;
			if (ptIdx == 0)
			{
				dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId].v[0]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];

				dev_primitives[pid + curPrimitiveBeginId + 2].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId + 2].v[1]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			}
			else if (ptIdx == 1)
			{
				dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId].v[0]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];

				dev_primitives[pid + curPrimitiveBeginId - 1].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId - 1].v[1]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			}
			else //if (ptIdx == 2)
			{
				dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId].v[0]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];

				dev_primitives[pid + curPrimitiveBeginId - 1].primitiveType = primitive.primitiveType;
				dev_primitives[pid + curPrimitiveBeginId - 1].v[1]
					= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			}
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_POINTS)
		{
			pid = iid;
			
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[0]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
	}
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__global__
void _rasterize(int width, int height, int numPrimitives, int * depth, Primitive* dev_primitives, Fragment *fragmentBuffer)
{
	// id for cur primitives vector
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < numPrimitives)
	{
		Primitive& primitive = dev_primitives[pid];
		if (primitive.primitiveType == PrimitiveType::Triangle)
		{
			const glm::vec3 triPixelPos[] =
			{ glm::vec3(primitive.v[0].pos), glm::vec3(primitive.v[1].pos), glm::vec3(primitive.v[2].pos) };

			AABB& triAabb = getAABBForTriangle(triPixelPos);
			triAabb.min.x = MIN( MAX(triAabb.min.x, 0), width);
			triAabb.min.y = MIN( MAX(triAabb.min.y, 0), height);

			triAabb.max.x = MIN( MAX(triAabb.max.x, 0), width);
			triAabb.max.y = MIN( MAX(triAabb.max.y, 0), height);

			for (size_t aabbX = triAabb.min.x; aabbX < triAabb.max.x + 1; aabbX++)
			{
				for (size_t aabbY = triAabb.min.y; aabbY < triAabb.max.y + 1; aabbY++)
				{
					glm::vec2 triAabbPos(aabbX, aabbY);
					glm::vec3 bCoeffs = calculateBarycentricCoordinate(triPixelPos, triAabbPos);
					if (isBarycentricCoordInBounds(bCoeffs))
					{
						const int pixelIdx = aabbX + aabbY * width;

						const float depthZ = getZAtCoordinate(bCoeffs, triPixelPos);
						const int depthBufferVal = INT_MAX * -depthZ; // depthZ will be between 0 (near) and -1 (far).
						atomicMin(&depth[pixelIdx], depthBufferVal);
						if (depth[pixelIdx] == depthBufferVal)
						{
							const VertexOut* triVtx = &primitive.v[0];
							Fragment& frag = fragmentBuffer[pixelIdx];

							frag.texWidth = primitive.texWidth;
							frag.texHeight = primitive.texHeight;
							frag.dev_diffuseTex = primitive.dev_diffuseTex;

							frag.eyeNor = bCoeffs[0] * triVtx[0].eyeNor
								+ bCoeffs[1] * triVtx[1].eyeNor
								+ bCoeffs[2] * triVtx[2].eyeNor;

							frag.eyePos = bCoeffs[0] * triVtx[0].eyePos
								+ bCoeffs[1] * triVtx[1].eyePos
								+ bCoeffs[2] * triVtx[2].eyePos;

							//frag.texcoord0	= bCoeffs[0] * triVtx[0].texcoord0
							//					+ bCoeffs[1] * triVtx[1].texcoord0
							//					+ bCoeffs[2] * triVtx[2].texcoord0;

							// Perspective-Correct Texturing.
							float perspZ = 1.0f / (bCoeffs[0] / primitive.v[0].pos.w
								+ bCoeffs[1] / primitive.v[1].pos.w
								+ bCoeffs[2] / primitive.v[2].pos.w);
							frag.color = Color(1.0f, 1.0f, 1.0f);
							frag.texcoord0 = bCoeffs[0] * triVtx[0].texcoord0 / primitive.v[0].pos.w
								+ bCoeffs[1] * triVtx[1].texcoord0 / primitive.v[1].pos.w
								+ bCoeffs[2] * triVtx[2].texcoord0 / primitive.v[2].pos.w;
							frag.texcoord0 *= perspZ;
						}
					}

				}
			}
		}
		else if (primitive.primitiveType == PrimitiveType::Line)
		{
			// Bresenham Algorithm
			// Taken from:
			// http://groups.csail.mit.edu/graphics/classes/6.837/F99/grading/asst2/turnin/rdror/Bresenham.java
			glm::vec2 vStart(primitive.v[0].pos);
			glm::vec2 vEnd(primitive.v[1].pos);
			
			vStart.x	= MIN( MAX( vStart.x, 0 ), width);
			vStart.y	= MIN( MAX( vStart.y, 0 ), height);
			
			vEnd.x		= MIN( MAX( vEnd.x, 0), width);
			vEnd.y		= MIN( MAX( vEnd.y, 0), height);

			float dx = vEnd.x - vStart.x; if (dx < 0) dx = -dx;
			float dy = vEnd.y - vStart.y; if (dy < 0) dy = -dy;

			int incx = 1; if (vEnd.x < vStart.x) incx = -1;
			int incy = 1; if (vEnd.y < vStart.y) incy = -1;

			int xStart = (int)(vStart.x + 0.5f); 
			int yStart = (int)(vStart.y + 0.5f);
			int xEnd = (int)(vEnd.x + 0.5f);
			int yEnd = (int)(vEnd.y + 0.5f);
			
			// If slope is outside the range [-1,1], swap x and y
			bool xy_swap = false;
			if (dy > dx) {
				xy_swap = true;
				int temp = xStart;
				xStart = yStart;
				yStart = temp;
				temp = xEnd;
				xEnd = yEnd;
				yEnd = temp;
			}

			// If line goes from right to left, swap the endpoints
			if (xEnd - xStart < 0) {
				int temp = xStart;
				xStart = xEnd;
				xEnd = temp;
				temp = yStart;
				yStart = yEnd;
				yEnd = temp;
			}

			int x,                       // Current x position
				y = yStart,                  // Current y position
				e = 0,                   // Current error
				m_num = yEnd - yStart,         // Numerator of slope
				m_denom = xEnd - xStart,       // Denominator of slope
				threshold = m_denom / 2;  // Threshold between E and NE increment 

			for (x = xStart; x < xEnd; x++) {
				if (xy_swap)
				{
					const int pixelIdx = y + x * width;
					fragmentBuffer[pixelIdx].color = Color(0.0f, 0.0f, 1.0f);
				}
				else
				{
					const int pixelIdx = x + y * width;
					fragmentBuffer[pixelIdx].color = Color(0.0f, 0.0f, 1.0f);
				}

				e += m_num;

				// Deal separately with lines sloping upward and those
				// sloping downward
				if (m_num < 0) {
					if (e < -threshold) {
						e += m_denom;
						y--;
					}
				}
				else if (e > threshold) {
					e -= m_denom;
					y++;
				}
			}

			if (xy_swap)
			{
				const int pixelIdx = y + x * width;
				fragmentBuffer[pixelIdx].color = Color(0.0f, 0.0f, 1.0f);
			}
			else
			{
				const int pixelIdx = x + y * width;
				fragmentBuffer[pixelIdx].color = Color(0.0f, 0.0f, 1.0f);
			}

		}
		else if (primitive.primitiveType == PrimitiveType::Point)
		{
			
			glm::vec2 vPoint(primitive.v[0].pos);
			if (0 < vPoint.x && vPoint.x < width && 0 < vPoint.y && vPoint.y < height)
			{
				static int i = 0;

				const int pixelIdx = ((int)vPoint.x) + ((int)vPoint.y) * width;
				fragmentBuffer[pixelIdx].color = Color(0.0f, 1.0f, 0.0f);
			}
		}
	}
}

// Taken from https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
__global__
void _buildSobelMap(int w, int h, int* dev_depth, Fragment *fragmentBuffer)
{
	const int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (0 < pixelX && pixelX < w && 0 < pixelY && pixelY < h)
	{
		const float filterX[3][3] = { { -1, -2, -1 }, { 0, 0,  0 }, { 1, 2, 1 } };
		const float filterY[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };

		float sobelX = 0.0f; float sobelY = 0.0f;
		for (int xOffset = -1; xOffset <= 1; xOffset += 1)
		{
			for (int yOffset = -1; yOffset <= 1; yOffset += 1)
			{
				if ((0 < pixelX + xOffset) && (pixelX + xOffset < w) && (0 < pixelY + yOffset) && (pixelY + yOffset < h))
				{
					const int pixelIdx = (pixelX + xOffset) + ((pixelY + yOffset) * w);
					const Fragment& frag = fragmentBuffer[pixelIdx];

					sobelX += filterX[xOffset + 1][yOffset + 1] * dev_depth[pixelIdx] / (1.0f*INT_MAX);
					sobelY += filterY[xOffset + 1][yOffset + 1] * dev_depth[pixelIdx] / (1.0f*INT_MAX);
				}
			}
		}
		const int pixelIdx = pixelX + pixelY * w;
		Fragment& frag = fragmentBuffer[pixelIdx];
		frag.sobelXY = sobelX * sobelX + sobelY * sobelY;
	}
}

__global__
void _buildSobelMapWithSharedMemory(int w, int h, int* dev_depth, Fragment *fragmentBuffer)
{
	const int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (0 < pixelX && pixelX < w && 0 < pixelY && pixelY < h)
	{
		const float filterX[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
		const float filterY[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };

		float sobelX = 0.0f; float sobelY = 0.0f;
		for (int xOffset = -1; xOffset <= 1; xOffset += 1)
		{
			for (int yOffset = -1; yOffset <= 1; yOffset += 1)
			{
				if ((0 < pixelX + xOffset) && (pixelX + xOffset < w) && (0 < pixelY + yOffset) && (pixelY + yOffset < h))
				{
					const int pixelIdx = (pixelX + xOffset) + ((pixelY + yOffset) * w);
					const Fragment& frag = fragmentBuffer[pixelIdx];

					sobelX += filterX[xOffset + 1][yOffset + 1] * dev_depth[pixelIdx] / (1.0f*INT_MAX);
					sobelY += filterY[xOffset + 1][yOffset + 1] * dev_depth[pixelIdx] / (1.0f*INT_MAX);
				}
			}
		}
		const int pixelIdx = pixelX + pixelY * w;
		Fragment& frag = fragmentBuffer[pixelIdx];
		frag.sobelXY = sobelX * sobelX + sobelY * sobelY;
	}
}

/**
* Writes fragment colors to the framebuffer
*/
__device__
Color tex2D(int w, int h, TextureData* tex2D, int texelX, int texelY)
{
	texelX = (texelX < w) ? texelX : w - 1;
	texelY = (texelY < h) ? texelY : h - 1;

	const int texelIdx = 3 * (texelX + texelY * w);
	if (tex2D != NULL)
	{
		TextureData* tex = &tex2D[texelIdx];
		return Color(tex[0] / 255.f, tex[1] / 255.f, tex[2] / 255.f);
	}
	else
	{
		return Color(0.0f, 0.0f, 1.0f);
	}
}

__device__
Color getTextureColor(const Fragment& frag)
{
	int texel[] = { frag.texcoord0.x * frag.texWidth, frag.texcoord0.y * frag.texHeight };

	TextureData* tex = frag.dev_diffuseTex;
	
	Color diffuseTexColor = tex2D(frag.texWidth, frag.texHeight, tex, texel[0], texel[1]);
	return diffuseTexColor;
}

__device__
Color getTextureBilinearColor(const Fragment& frag)
{
	float texeluv[]		= { frag.texcoord0.x * frag.texWidth - 0.5, frag.texcoord0.y * frag.texHeight - 0.5 };
	int texelxy[]		= { floor(texeluv[0]), floor(texeluv[1]) };

	float texel_diff[]	= { texeluv[0] - texelxy[0], texeluv[1] - texelxy[1] };
	float texel_OppDiff[] = { 1 - texel_diff[0], 1 - texel_diff[1] };

	TextureData* tex = frag.dev_diffuseTex;
	Color tex00 = tex2D(frag.texWidth, frag.texHeight, tex,	  texelxy[0],	texelxy[1]);
	Color tex10 = tex2D(frag.texWidth, frag.texHeight, tex, texelxy[0]+1,	texelxy[1]);
	Color tex01 = tex2D(frag.texWidth, frag.texHeight, tex,   texelxy[0],	texelxy[1]+1);
	Color tex11 = tex2D(frag.texWidth, frag.texHeight, tex, texelxy[0]+1,	texelxy[1]+1);

	Color result = (texel_OppDiff[0] * tex00 + texel_diff[0] * tex10) * texel_OppDiff[1]
				+ (texel_OppDiff[0] * tex01 + texel_diff[0] * tex11) * texel_diff[1];
	return result;
}

__global__
void render(int w, int h, int* depth, Fragment *fragmentBuffer, Color *framebuffer)
{
	const glm::vec3 lightPos		= glm::vec3(0, 5, 0);
	const Color matAmbientClr		= Color(0.25f, 0.25f, 0.25f);
	const glm::vec3 matDiffuseClr	= Color(0.3f, 0.3f, 0.3f);
	const float lightDiffuseI		= 0.65f;
	const glm::vec3 matSpecColor	= Color(0.03f, 0.03f, 0.03f);
	const float matSpecShininess	= 2.5f;
	const float lightSpecI			= 0.75f;
	
	const float toonShadingVariance = 12;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pixelIdx = x + (y * w);
	
	if (0 < x && x < w && 0 < y && y < h)
	{
		const Fragment& frag = fragmentBuffer[pixelIdx];
		Color& pixelClr = framebuffer[pixelIdx];
		{
			if (depth[pixelIdx] != INT_MAX && frag.dev_diffuseTex)
			{
				glm::vec3 unitEyeNormal = glm::normalize(frag.eyeNor);
				glm::vec3 lightVec = glm::normalize(lightPos - frag.eyePos);
				float diffCoeff = MIN(MAX(0, glm::dot(lightVec, unitEyeNormal)), 1.0f);

				glm::vec3 lightReflV = -glm::normalize(glm::reflect(lightVec, unitEyeNormal));
				float specCoeff = MIN( MAX(0.0f, glm::dot(lightReflV, glm::normalize(-frag.eyePos))), 1.0f);
				specCoeff = pow(specCoeff, matSpecShininess);

				//pixelClr = getTextureColor(frag);
				pixelClr	= matAmbientClr 
							+ lightDiffuseI * diffCoeff * getTextureBilinearColor(frag)
							+ lightSpecI * specCoeff * matSpecColor;

				// The cel - shading process starts with a typical 3D model. Where cel-shading differs
				// from conventional rendering is in its non - photorealistic illumination model.
				// Conventional(smooth) lighting values are calculated for each pixel and then quantized
				//to a small number of discrete shades to create the characteristic flat look:
				pixelClr = glm::ceil(pixelClr * toonShadingVariance) / toonShadingVariance;
				//Black "ink" outlines and contour lines can be created using a variety of methods.
				// We use the Sobel method to create a shared memory buffer.
				if (frag.sobelXY > 1.0f) pixelClr = glm::vec3(0.0f, 0.0f, 0.0f);
			}
			else
			{
				pixelClr = frag.color;
			}
		}
	}
}

/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, Color *image) {
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

struct IsPrimitiveBackFacing
{
	__host__ __device__
	bool operator()(const Primitive& p)
	{
		switch (p.primitiveType)
		{
			case PrimitiveType::Line:
			{
				glm::vec3 frontFacingDegree = 0.5f * glm::vec3(p.v[1].eyeNor + p.v[0].eyeNor);
				return frontFacingDegree.z < 0;
			}
			break;
			case PrimitiveType::Triangle:
			{
				glm::vec3 normal = glm::normalize(glm::cross(
				glm::vec3(p.v[1].eyePos - p.v[0].eyePos),
				glm::vec3(p.v[2].eyePos - p.v[0].eyePos)));

				float frontFacingDegree = glm::dot(glm::vec3(-p.v[0].eyePos), normal);
				return frontFacingDegree < 0;
			}
			break;
			case PrimitiveType::Point:
			default:
				return false;
		}
	}
};

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1, (height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)
	Timer::resetTimer(true);
	Timer::playTimer();
	{
		cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
		initFrameBuffers << <blockCount2d, blockSize2d >> >(width, height, dev_depth, dev_fragmentBuffer);
	}
	Timer::pauseTimer();
	Timer::printTimer("InitFB: ", 1.0f);

	// Vertex Process & primitive assembly
	Timer::resetTimer(true);
	Timer::playTimer();
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
	Timer::pauseTimer();
	Timer::printTimer("VP: ", 1.0f);

	//Back-Face culling
	Timer::resetTimer(true);
	Timer::playTimer();
	{
		totalNumPrimitives = curPrimitiveBeginId;
		//printf("Pre-Cull: %d;", totalNumPrimitives);
		cudaMalloc(&dev_primitivesCulled, totalNumPrimitives * sizeof(Primitive));
		cudaMemcpy(dev_primitivesCulled, dev_primitives, totalNumPrimitives * sizeof(Primitive), cudaMemcpyDeviceToDevice);

		thrust::device_ptr<Primitive> dPtr_primitivesBegin(dev_primitivesCulled);
		thrust::device_ptr<Primitive> dPtr_primitivesEnd = dPtr_primitivesBegin + totalNumPrimitives;
		dPtr_primitivesEnd = thrust::remove_if(dPtr_primitivesBegin, dPtr_primitivesEnd, IsPrimitiveBackFacing());
		totalNumPrimitives = dPtr_primitivesEnd - dPtr_primitivesBegin;
		checkCUDAError("backface culling");

		//printf(" Post-Cull: %d\n", totalNumPrimitives);
	}
	Timer::pauseTimer();
	Timer::printTimer("BackFace: ", 1.0f);

	if (totalNumPrimitives > 0)
	{
		// TODO: rasterize
		Timer::resetTimer(true);
		Timer::playTimer();
		{
			dim3 numThreadsPerBlock(256);
			dim3 numBlocksForPrims((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

			_rasterize << < numBlocksForPrims, numThreadsPerBlock >> > (width, height, totalNumPrimitives, dev_depth, dev_primitivesCulled, dev_fragmentBuffer);
		}
		Timer::pauseTimer();
		Timer::printTimer("Rasterize: ", 1.0f);

		// Sobel Filter
		Timer::resetTimer(true);
		Timer::playTimer();
		{
			dim3 numThreadsPerBlock(SOBEL_GRID_SIZE, SOBEL_GRID_SIZE, 1);
			dim3 numBlocksForSobel((width - 1) / numThreadsPerBlock.x + 1, (height - 1) / numThreadsPerBlock.y + 1);
			_buildSobelMap << < numBlocksForSobel, numThreadsPerBlock >> > (width, height, dev_depth, dev_fragmentBuffer);
		}
		Timer::pauseTimer();
		Timer::printTimer("SobelMap: ", 1.0f);
	}
	
	Timer::resetTimer(true);
	Timer::playTimer();
	{
		// Copy depthbuffer colors into framebuffer
		render << <blockCount2d, blockSize2d >> >(width, height, dev_depth, dev_fragmentBuffer, dev_framebuffer);
		checkCUDAError("fragment shader");
	}
	Timer::pauseTimer();
	Timer::printTimer("FS: ", 1.0f);

	Timer::resetTimer(true);
	Timer::playTimer();
	{
		// Copy framebuffer into OpenGL buffer for OpenGL previewing
		sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
		checkCUDAError("copy render result to pbo");
	}
	Timer::pauseTimer();
	Timer::printTimer("ToOGL: ", 1.0f);
	cudaFree(dev_primitivesCulled);
}
