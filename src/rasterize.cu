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

#define BILINEAR_INTERP 1
#define PERSPECTIVE_CORRECT 1
#define BLINN_PHONG_SHADING 1 

#define DiffuseColor glm::vec3(0.1, 0.1, 0.1)

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
		// TextureData* dev_diffuseTex = NULL;
		// int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];

		// texture
		TextureData* dev_diffuseTex = NULL;
		int diffuseTexWidth, diffuseTexHeight;
		int diffuseTexStride; // RGB RGBA
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;

		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int diffuseTexWidth, diffuseTexHeight;
		int diffuseTexStride; // RGB RGBA
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
		int diffuseTexWidth, diffuseTexHeight;
		int diffuseTexStride; 

		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

//since atomicMin only supports int?
static int * dev_depth = NULL;	// you might need this buffer when doing depth test


//raterization mode
RASTERIZATION_MODE rasterization_mode;

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
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

//__global__
//void initDepth(int w, int h, int * depth)
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
					TextureData* dev_diffuseTex = NULL;
					int texWidth, texHeight, texStride;

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
									
									// TODO: store the image size to your PrimitiveDevBufPointers
									texWidth = image.width;
									texHeight = image.height;
									texStride = image.component;

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

						dev_diffuseTex,
						texWidth,
						texHeight,
						texStride,

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

		// DONE: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		glm::vec4 pos4 = glm::vec4(primitive.dev_position[vid], 1.0f);

		// into clipping space
		glm::vec4 posOut = MVP * pos4;

		// into NDC space(-1,1)
		posOut = posOut / posOut.w;

		// into view port space and z value should be the depth
		// posOut will be in pixel sapce (width, height),
		posOut.x = (1.0 - posOut.x) * 0.5 * width;  // x is column index
		posOut.y = (1.0 - posOut.y) * 0.5 * height; // y is row index


		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].pos = posOut;  
		primitive.dev_verticesOut[vid].eyePos = multiplyMV(MV, pos4);
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		
		if (primitive.dev_texcoord0 != NULL)
		{
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		}
		else
		{
			primitive.dev_verticesOut[vid].texcoord0 = glm::vec2(-1, -1);
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

			// set texture informations
			dev_primitives[pid + curPrimitiveBeginId].dev_diffuseTex = primitive.dev_diffuseTex;
			dev_primitives[pid + curPrimitiveBeginId].diffuseTexWidth = primitive.diffuseTexWidth;
			dev_primitives[pid + curPrimitiveBeginId].diffuseTexHeight = primitive.diffuseTexHeight;
			dev_primitives[pid + curPrimitiveBeginId].diffuseTexStride = primitive.diffuseTexStride;

		}


		// TODO: other primitive types (point, line)
	}
	
}


/**
 * Rasterization kernel func.
 * Perspective correct reference:
 * https://www.comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf
 */
__global__ 
void _rasterizationSolidMode(
	int numPrimitives,
	Primitive* dev_primitives,
	int * dev_depth,
	Fragment * dev_fragmentBuffer,
	int width, int height)
{

	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx >= numPrimitives)
		return;

	//TODO: linear inerpolation for all attributes except pos

	Primitive & primitive = dev_primitives[primitiveIdx];

	// set up for barycentric coordinates interpolation
	// get 3 vertex pos 
	glm::vec3 tri_pos[3] = {
		glm::vec3(primitive.v[0].pos),
		glm::vec3(primitive.v[1].pos),
		glm::vec3(primitive.v[2].pos)
	};

	

#if PERSPECTIVE_CORRECT == 1
	// get z value of eye space position
	float z_eyePos[3] = {
		primitive.v[0].eyePos.z,
		primitive.v[1].eyePos.z,
		primitive.v[2].eyePos.z
	};

#endif


	// get AABB for this primitive
	AABB aabb = getAABBForTriangle(tri_pos);

	// for each pixel inside aabb, fill in a color
	glm::vec3 baryCoord;
	float depth;
	int pixelIdx;

	for (int x = glm::max(0, (int)aabb.min.x); x <= glm::min(width - 1, (int)aabb.max.x); ++x)
	{
		for (int y = glm::max(0, (int)aabb.min.y); y <= glm::min(height -1, (int)aabb.max.y); ++y)
		{
			// get barycentrix coordinates
			baryCoord = calculateBarycentricCoordinate(tri_pos, glm::vec2(x, y));

			// if is inside the primitive
			if (isBarycentricCoordInBounds(baryCoord))
			{
				if (x < 0 || x >= width || y < 0 || y >= height)
					continue;

				depth = getZAtCoordinate(baryCoord, tri_pos);

				pixelIdx = x + y * width; // pixel index
				
				int d = INT_MAX * -depth; // convert to int, 
				atomicMin(&dev_depth[pixelIdx], d);
				if (dev_depth[pixelIdx] == d)
				{
					
#if PERSPECTIVE_CORRECT == 1

					float correctZ = getZAtCoordinatePerspectiveCorrect(baryCoord, z_eyePos);
					dev_fragmentBuffer[pixelIdx].eyePos = correctZ * (
						baryCoord.x * primitive.v[0].eyePos / z_eyePos[0] +
						baryCoord.y * primitive.v[1].eyePos / z_eyePos[1] +
						baryCoord.z * primitive.v[2].eyePos / z_eyePos[2] );
						
					dev_fragmentBuffer[pixelIdx].eyeNor = correctZ * (
						baryCoord.x * primitive.v[0].eyeNor / z_eyePos[0] +
						baryCoord.y * primitive.v[1].eyeNor / z_eyePos[1] +
						baryCoord.z * primitive.v[2].eyeNor / z_eyePos[2] );
					
					dev_fragmentBuffer[pixelIdx].texcoord0 = correctZ *(
						baryCoord.x * primitive.v[0].texcoord0 / z_eyePos[0] +
						baryCoord.y * primitive.v[1].texcoord0 / z_eyePos[1] +
						baryCoord.z * primitive.v[2].texcoord0 / z_eyePos[2] );

#else 

					dev_fragmentBuffer[pixelIdx].eyePos = 
						baryCoord.x * primitive.v[0].eyePos +
						baryCoord.y * primitive.v[1].eyePos +
						baryCoord.z * primitive.v[2].eyePos ;

					dev_fragmentBuffer[pixelIdx].eyeNor = 
						baryCoord.x * primitive.v[0].eyeNor +
						baryCoord.y * primitive.v[1].eyeNor +
						baryCoord.z * primitive.v[2].eyeNor ;

					dev_fragmentBuffer[pixelIdx].texcoord0 = 
						baryCoord.x * primitive.v[0].texcoord0 +
						baryCoord.y * primitive.v[1].texcoord0 +
						baryCoord.z * primitive.v[2].texcoord0 ;

#endif

					dev_fragmentBuffer[pixelIdx].color = DiffuseColor; // black
					dev_fragmentBuffer[pixelIdx].dev_diffuseTex = primitive.dev_diffuseTex;
					dev_fragmentBuffer[pixelIdx].diffuseTexWidth = primitive.diffuseTexWidth;
					dev_fragmentBuffer[pixelIdx].diffuseTexHeight = primitive.diffuseTexHeight;
					dev_fragmentBuffer[pixelIdx].diffuseTexStride = primitive.diffuseTexStride;

				}
			}
			
		}
	}

}

/**
*	point mode rasterization
*/
__global__
void _rasterizationPointMode(
	int numPrimitives,
	Primitive* dev_primitives,
	int * dev_depth,
	Fragment * dev_fragmentBuffer,
	int width, int height)
{
	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx >= numPrimitives)
		return;

	Primitive & primitive = dev_primitives[primitiveIdx];

	for (int i = 0; i < 3; i++)
	{
		int x = primitive.v[i].pos.x;
		int y = primitive.v[i].pos.y;
		if (x >= 0 && x < width && y >= 0 && y < height)
		{
			int depth = INT_MAX * primitive.v[i].pos.z;
			int pixelIdx = x + y * width;

			atomicMin(&dev_depth[pixelIdx], depth);

			if (dev_depth[pixelIdx] == depth)
			{
				dev_fragmentBuffer[pixelIdx].color = DiffuseColor;
				dev_fragmentBuffer[pixelIdx].eyePos = primitive.v[i].eyePos;
				dev_fragmentBuffer[pixelIdx].eyeNor = primitive.v[i].eyeNor;
				dev_fragmentBuffer[pixelIdx].texcoord0 = primitive.v[i].texcoord0;

				dev_fragmentBuffer[pixelIdx].dev_diffuseTex = primitive.dev_diffuseTex;
				dev_fragmentBuffer[pixelIdx].diffuseTexWidth = primitive.diffuseTexWidth;
				dev_fragmentBuffer[pixelIdx].diffuseTexHeight = primitive.diffuseTexHeight;
				dev_fragmentBuffer[pixelIdx].diffuseTexStride = primitive.diffuseTexStride;
			}

		}
	}
}

/**
* wireframe mode rasterization
*/
__global__
void _rasterizationWireframeMode(
	int numPrimitives,
	Primitive* dev_primitives,
	int * dev_depth,
	Fragment * dev_fragmentBuffer,
	int width, int height)
{
	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx >= numPrimitives)
		return;

	Primitive & primitive = dev_primitives[primitiveIdx];

	for (int i = 0; i < 3; i++)
	{
		VertexOut & startPoint = primitive.v[i];
		VertexOut & endPoint = primitive.v[(i + 1) % 3];

		int xRange = (int)endPoint.pos.x - (int)startPoint.pos.x;
		int yRange = (int)endPoint.pos.y - (int)startPoint.pos.y;
		int dx = xRange > 0 ? 1 : -1;
		int dy = yRange > 0 ? 1 : -1;

		if (glm::abs(xRange) > glm::abs(yRange)) // x - major fill
		{
			int x = startPoint.pos.x;
			int y;
			int targetX = (int)endPoint.pos.x;

			while (x != targetX)
			{
				float ratio = glm::abs(float(x) - startPoint.pos.x) / float(glm::abs(xRange));
				y =  ratio * yRange + startPoint.pos.y;

				if (x >= 0 && x < width && y >= 0 && y < height)
				{
					int depth = INT_MAX * ((1 - ratio) * startPoint.pos.z + ratio * endPoint.pos.z);
					int pixelIdx = x + y * width;

					atomicMin(&dev_depth[pixelIdx], depth);

					if (dev_depth[pixelIdx] == depth)
					{
						dev_fragmentBuffer[pixelIdx].color = DiffuseColor;
						dev_fragmentBuffer[pixelIdx].eyePos = (1 - ratio) * startPoint.eyePos + ratio * endPoint.eyePos;
						dev_fragmentBuffer[pixelIdx].eyeNor = (1 - ratio) * startPoint.eyeNor + ratio * endPoint.eyeNor;
						dev_fragmentBuffer[pixelIdx].texcoord0 = (1 - ratio) * startPoint.texcoord0 + ratio * endPoint.texcoord0;

						dev_fragmentBuffer[pixelIdx].dev_diffuseTex = primitive.dev_diffuseTex;
						dev_fragmentBuffer[pixelIdx].diffuseTexWidth = primitive.diffuseTexWidth;
						dev_fragmentBuffer[pixelIdx].diffuseTexHeight = primitive.diffuseTexHeight;
						dev_fragmentBuffer[pixelIdx].diffuseTexStride = primitive.diffuseTexStride;
					}
				}

				x += dx;
			}
		}
		else // y - major
		{
			int y = startPoint.pos.y;
			int x;
			int targetY = (int)endPoint.pos.y;

			while (y != targetY)
			{
				float ratio = glm::abs(float(y) - startPoint.pos.y) / float(glm::abs(yRange));
				x = ratio * xRange + startPoint.pos.x;

				if (x >= 0 && x < width && y >= 0 && y < height)
				{
					int depth = INT_MAX * ((1 - ratio) * startPoint.pos.z + ratio * endPoint.pos.z);
					int pixelIdx = x + y * width;

					atomicMin(&dev_depth[pixelIdx], depth);

					if (dev_depth[pixelIdx] == depth)
					{
						dev_fragmentBuffer[pixelIdx].color = DiffuseColor;
						dev_fragmentBuffer[pixelIdx].eyePos = (1 - ratio) * startPoint.eyePos + ratio * endPoint.eyePos;
						dev_fragmentBuffer[pixelIdx].eyeNor = (1 - ratio) * startPoint.eyeNor + ratio * endPoint.eyeNor;
						dev_fragmentBuffer[pixelIdx].texcoord0 = (1 - ratio) * startPoint.texcoord0 + ratio * endPoint.texcoord0;

						dev_fragmentBuffer[pixelIdx].dev_diffuseTex = primitive.dev_diffuseTex;
						dev_fragmentBuffer[pixelIdx].diffuseTexWidth = primitive.diffuseTexWidth;
						dev_fragmentBuffer[pixelIdx].diffuseTexHeight = primitive.diffuseTexHeight;
						dev_fragmentBuffer[pixelIdx].diffuseTexStride = primitive.diffuseTexStride;
					}
				}

				y += dy;
			}
		}

	}

}

/*
* helper function for get pixel color from texture file
*/
__device__
glm::vec3 getTextureColor(
	TextureData* texture,
	int texWidth, int texHeight,
	int texStride,
	int x, int y)
{
	int texPixelIdx = texStride * (x + y * texWidth);

	if (x >= 0 && y >= 0 && x < texWidth && y < texHeight && texture != NULL)
	{
		return glm::vec3(
			(float)texture[texPixelIdx + 0] / 255.0f,
			(float)texture[texPixelIdx + 1] / 255.0f,
			(float)texture[texPixelIdx + 2] / 255.0f);
	}
	else
	{
		return glm::vec3(0, 0, 0);
	}
}

/**
* Writes fragment colors to the framebuffer
* bilinear filter: reference
* https://en.wikipedia.org/wiki/Bilinear_filtering
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		//framebuffer[index] = fragmentBuffer[index].eyeNor;

		// TODO: add your fragment shader code here
		Fragment & frag = fragmentBuffer[index];

#if BILINEAR_INTERP == 1


		if (frag.dev_diffuseTex == NULL)
		{
			framebuffer[index] = frag.color;
		}
		else
		{
			float fx = frag.texcoord0.x * frag.diffuseTexWidth;
			float fy = frag.texcoord0.y * frag.diffuseTexHeight;
			int tx = glm::max(0, glm::min(int(fx), frag.diffuseTexWidth - 1));
			int ty = glm::max(0, glm::min(int(fy), frag.diffuseTexHeight - 1));

			float dx = fx - tx;
			float dy = fy - ty;

			glm::vec3 tex_x_y = getTextureColor(
				frag.dev_diffuseTex,
				frag.diffuseTexWidth, frag.diffuseTexHeight, frag.diffuseTexStride,
				tx, ty);
			glm::vec3 tex_x_1_y = getTextureColor(
				frag.dev_diffuseTex,
				frag.diffuseTexWidth, frag.diffuseTexHeight, frag.diffuseTexStride,
				tx + 1, ty);
			glm::vec3 tex_x_y_1 = getTextureColor(
				frag.dev_diffuseTex,
				frag.diffuseTexWidth, frag.diffuseTexHeight, frag.diffuseTexStride,
				tx, ty + 1);
			glm::vec3 tex_x_1_y_1 = getTextureColor(
				frag.dev_diffuseTex,
				frag.diffuseTexWidth, frag.diffuseTexHeight, frag.diffuseTexStride,
				tx + 1, ty + 1);

			framebuffer[index] =
				(tex_x_y * (1 - dx) + tex_x_1_y * dx) * (1 - dy) +
				(tex_x_y_1 * (1 - dx) + tex_x_1_y_1 * dx) * dy;
		}


#else

		if (frag.dev_diffuseTex == NULL)
		{
			framebuffer[index] = frag.color;
		}
		else
		{
			int tx = frag.texcoord0.x * frag.diffuseTexWidth; 
			int ty = frag.texcoord0.y * frag.diffuseTexHeight;
			int texPixelIdx = frag.diffuseTexStride * (tx + ty * frag.diffuseTexWidth);

			TextureData * texture = frag.dev_diffuseTex;
			if ( tx >= 0 && ty >=0 && tx < frag.diffuseTexWidth && ty < frag.diffuseTexHeight && texture != NULL)
			{
				framebuffer[index] = glm::vec3(
					(float)texture[texPixelIdx + 0] / 255.0f,
					(float)texture[texPixelIdx + 1] / 255.0f,
					(float)texture[texPixelIdx + 2] / 255.0f );
			}
			else
			{
				framebuffer[index] = frag.color;
			}
		}

#endif

#if BLINN_PHONG_SHADING == 1
		// lighting shading Blinn - Phong model
		// reference: https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model
		glm::vec3 lightDir = glm::normalize(glm::vec3(0, 1, 0) - frag.eyePos);
		float lambert = glm::max(glm::dot(lightDir, frag.eyeNor), 0.0f);

		float specular = 0.0f;

		if (lambert > 0);
		{
			glm::vec3 viewDir = glm::normalize(-frag.eyePos);

			glm::vec3 halfDir = glm::normalize(lightDir + viewDir);
			float specAngle = glm::max(glm::dot(halfDir, frag.eyeNor), 0.0f);
			specular = powf(specAngle, 24.0);
		}

		// ambient + lambert + specular
		glm::vec3 color = 
			//glm::vec3(0.1, 0.1, 0.1)	+	
			lambert * framebuffer[index] +
			specular * glm::vec3(1, 1, 1);		

		color.x = powf(color.x, 1.0f / 2.2f);
		color.y = powf(color.y, 1.0f / 2.2f);
		color.z = powf(color.z, 1.0f / 2.2f);
		framebuffer[index] = color;
	}

#endif

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
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// TODO: rasterize
	{
		int threadsPerBlock = 128;
		dim3 numThreadsPerBlock(threadsPerBlock);
		dim3 numBlocksForPrimitives((totalNumPrimitives + threadsPerBlock - 1) / threadsPerBlock);

		switch (rasterization_mode)
		{
		case RASTERIZATION_MODE::Point:
			_rasterizationPointMode << <numBlocksForPrimitives, numThreadsPerBlock >> >(
				totalNumPrimitives,
				dev_primitives,
				dev_depth,
				dev_fragmentBuffer,
				width,
				height);
			break;

		case RASTERIZATION_MODE::Wireframe:
			_rasterizationWireframeMode << <numBlocksForPrimitives, numThreadsPerBlock >> >(
				totalNumPrimitives,
				dev_primitives,
				dev_depth,
				dev_fragmentBuffer,
				width,
				height);
			break;

		case RASTERIZATION_MODE::Solid:
			_rasterizationSolidMode << <numBlocksForPrimitives, numThreadsPerBlock >> >(
				totalNumPrimitives,
				dev_primitives,
				dev_depth,
				dev_fragmentBuffer,
				width,
				height);
			break;

		default:
			break;
		}
		
		checkCUDAError("rasterization");
	}


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

    checkCUDAError("rasterize Free");
}
