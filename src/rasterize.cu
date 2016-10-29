/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
    cudaArray* dev_diffuseTex = NULL;
    cudaTextureObject_t dev_diffuseTexObj = 0;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
    glm::vec3 eyePos;
    glm::vec3 eyeNor;
    float z, ssao;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful,
		// but always feel free to modify on your own

		// glm::vec3 eyePos;	// eye space position used for shading
		// glm::vec3 eyeNor;
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
		cudaArray *dev_diffuseTex;
    cudaTextureObject_t dev_diffuseTexObj;
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

static unsigned long long *dev_depth = NULL;

static glm::ivec2 *dev_tiles_min = NULL, *dev_tiles_max = NULL;

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

static cudaTextureObject_t dev_ssaoTexObj = 0;
static cudaArray *dev_ssaoTexArray = nullptr;
static glm::vec3 *dev_ssaoKernel = nullptr;
static cudaArray *dev_ssaoOutArray = nullptr;
static cudaSurfaceObject_t dev_ssaoOutSurfObj;
static cudaTextureObject_t dev_ssaoOutTexObj;
void ssaoInit(int nKernel, int nRandom, int width, int height) {
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::default_random_engine rng;

  // generate sampling kernel
  glm::vec3 *kern = new glm::vec3[nKernel*nKernel];
  float scale = 1.0f / (nKernel*nKernel);
  for (int i = 0; i < nKernel*nKernel; i++) {
    kern[i].x = 2.0f*dist(rng) - 1.0f;
    kern[i].y = 2.0f*dist(rng) - 1.0f;
    kern[i].z = dist(rng);
    kern[i] = glm::normalize(kern[i]) * scale;
    scale = 0.1f + 0.9f*scale*scale;
  }
  cudaMalloc(&dev_ssaoKernel, nKernel*nKernel*sizeof(glm::vec3));
  cudaMemcpy(dev_ssaoKernel, kern, nKernel*nKernel*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  delete kern;

  // generate randomization texture
  float4 *noise = new float4[nRandom*nRandom];
  for (int i = 0; i < nRandom*nRandom; i++) {
    noise[i].x = 2.0f*dist(rng) - 1.0f;
    noise[i].y = 2.0f*dist(rng) - 1.0f;
    noise[i].z = noise[i].w = 0.0f;
  }
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<float4>();
  cudaMallocArray(&dev_ssaoTexArray, &channel, nRandom, nRandom);
  cudaMemcpyToArray(dev_ssaoTexArray, 0, 0, noise, nRandom*nRandom*sizeof(float4), cudaMemcpyHostToDevice);
  checkCUDAError("Set Texture Image data");
  delete noise;

  // Specify texture
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = dev_ssaoTexArray;

  // Specify texture object parameters
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeWrap;
  texDesc.addressMode[1]   = cudaAddressModeWrap;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaCreateTextureObject(&dev_ssaoTexObj, &resDesc, &texDesc, NULL);

  // Create output array
  channel = cudaCreateChannelDesc<float>();
  cudaMallocArray(&dev_ssaoOutArray, &channel, width, height);
  resDesc.res.array.array = dev_ssaoOutArray;
  cudaCreateSurfaceObject(&dev_ssaoOutSurfObj, &resDesc);
  cudaCreateTextureObject(&dev_ssaoOutTexObj, &resDesc, &texDesc, NULL);
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
	cudaMalloc(&dev_depth, width * height * sizeof(unsigned long long));

  ssaoInit(8, 4, w, h);

	checkCUDAError("rasterizeInit");
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
					VertexIndex* dev_indices = nullptr;
					VertexAttributePosition* dev_position = nullptr;
					VertexAttributeNormal* dev_normal = nullptr;
					VertexAttributeTexcoord* dev_texcoord0 = nullptr;

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
					cudaArray* dev_diffuseTex = NULL;
          cudaTextureObject_t dev_diffuseTexObj = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

                  // convert image to rgba
                  std::vector<TextureData> rgbaImg;
                  for (int i = 0; i < image.image.size(); i++) {
                    rgbaImg.push_back(image.image[i]);
                    if(i % 3 == 2)
                      rgbaImg.push_back(0);
                  }

									size_t s =rgbaImg.size() * sizeof(TextureData);
                  //printf("img size %d\n", s);
                  cudaChannelFormatDesc channel = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
									cudaMallocArray(&dev_diffuseTex, &channel, image.width, image.height);
									cudaMemcpyToArray(dev_diffuseTex, 0, 0, &rgbaImg[0], s, cudaMemcpyHostToDevice);
									checkCUDAError("Set Texture Image data");

                  // Specify texture
                  cudaResourceDesc resDesc;
                  memset(&resDesc, 0, sizeof(resDesc));
                  resDesc.resType = cudaResourceTypeArray;
                  resDesc.res.array.array = dev_diffuseTex;

                  // Specify texture object parameters
                  cudaTextureDesc texDesc;
                  memset(&texDesc, 0, sizeof(texDesc));
                  texDesc.addressMode[0]   = cudaAddressModeWrap;
                  texDesc.addressMode[1]   = cudaAddressModeWrap;
                  texDesc.filterMode       = cudaFilterModeLinear;
                  texDesc.readMode         = cudaReadModeNormalizedFloat;
                  texDesc.normalizedCoords = 1;

                  // Create texture object
                  cudaCreateTextureObject(&dev_diffuseTexObj, &resDesc, &texDesc, NULL);

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
            dev_diffuseTexObj,

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
    cudaMalloc(&dev_tiles_min, totalNumPrimitives * sizeof(glm::ivec2));
    cudaMalloc(&dev_tiles_max, totalNumPrimitives * sizeof(glm::ivec2));
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

    VertexOut &vOut = primitive.dev_verticesOut[vid];

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

    vOut.pos = MVP*glm::vec4(primitive.dev_position[vid], 1);
    vOut.pos.x /= vOut.pos.w;
    vOut.pos.y /= vOut.pos.w;
    vOut.pos.z /= vOut.pos.w;
    vOut.pos.x = 0.5f * width * (vOut.pos.x + 1.0f);
    vOut.pos.y = 0.5f * height * (1.0f - vOut.pos.y);

    vOut.eyePos = glm::vec3(MV*glm::vec4(primitive.dev_position[vid], 1));
    vOut.eyeNor = glm::normalize(MV_normal*primitive.dev_normal[vid]);

    vOut.dev_diffuseTex = primitive.dev_diffuseTex;
    vOut.dev_diffuseTexObj = primitive.dev_diffuseTexObj;

    if (primitive.dev_texcoord0)
      vOut.texcoord0 = primitive.dev_texcoord0[vid];

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

// based on http://stereopsis.com/radix.html
// maps float to uint32 in a way that preserves order. stores in the upper half
// of a uint64 , and the index of a primitive as the lower 32, so atomicMin can both
// compare based on depth and store the index
__device__ static inline uint64_t FloatFlip(float f, int i)
{
	uint32_t fi = *((uint32_t*)&f), mask = (fi & 0x80000000) ? 0xFFFFFFFF : 0x80000000;
	return (((uint64_t)(fi ^ mask)) << 32) | i;
}
__device__ static inline float FloatUnflip(uint64_t u)
{
  u >>= 32;
  uint32_t mask = (u & 0x80000000) ? 0xFFFFFFFF : 0x80000000, fi = u ^ mask;
	return *((float*)&fi);
}

__global__ void depthPass(int numPrimitives, Primitive *dev_primitives, int w, int h, unsigned long long *depth) {
  int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pIdx >= numPrimitives)
    return;
  Primitive &p = dev_primitives[pIdx];

  glm::vec3 tri[3];
  tri[0] = glm::vec3(p.v[0].pos);
  tri[1] = glm::vec3(p.v[1].pos);
  tri[2] = glm::vec3(p.v[2].pos);

  AABB aabb;
  getAABBForTriangle(tri, aabb);
  for (int j = aabb.min[1]; j < aabb.max[1] && j < h && j >= 0; j++) {
  for (int i = aabb.min[0]; i < aabb.max[0] && i < w && i >= 0; i++) {
    glm::vec2 coord(i,j);
    glm::vec3 bary = calculateBarycentricCoordinate(tri, coord);
    if (isBarycentricCoordInBounds(bary))
      atomicMin(&depth[i+w*j], FloatFlip(getZAtCoordinate(bary, tri),pIdx));
  }}
}


__device__ void shadeFragment(unsigned int i, unsigned int j, const Primitive &p, glm::vec3 &out) {
  glm::vec3 tri[3];
  tri[0] = glm::vec3(p.v[0].pos);
  tri[1] = glm::vec3(p.v[1].pos);
  tri[2] = glm::vec3(p.v[2].pos);
  glm::vec2 coord(i,j);
  glm::vec3 bary = calculateBarycentricCoordinate(tri, coord);

  // lambert
  glm::vec3 fragDir = glm::normalize(bary.x*p.v[0].eyePos + bary.y*p.v[1].eyePos + bary.z*p.v[2].eyePos);
  glm::vec3 fragNrm = glm::normalize(bary.x*p.v[0].eyeNor + bary.y*p.v[1].eyeNor + bary.z*p.v[2].eyeNor);
  glm::vec3 lambert = glm::clamp(-glm::vec3(glm::dot(fragDir, fragNrm)), 0.0f, 1.0f);

  glm::vec3 texBary = bary / glm::vec3(p.v[0].pos[3], p.v[1].pos[3], p.v[2].pos[3]);

  glm::vec2 st = texBary[0]*p.v[0].texcoord0 + texBary[1]*p.v[1].texcoord0 + texBary[2]*p.v[2].texcoord0;
  float norm = texBary[0] + texBary[1] + texBary[2];
  if (p.v[0].dev_diffuseTex) {
    float4 rgba = tex2D<float4>(p.v[0].dev_diffuseTexObj, st.x / norm, st.y / norm);
    lambert *= glm::vec3(rgba.x,rgba.y,rgba.z);
  }


  out = lambert;
}

__global__ void _fragRasterize(int numPrimitives, Primitive *dev_primitives, Fragment *dev_fragments, int w, int h, unsigned long long *depth, glm::vec3 *framebuffer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= w || j >= h)
    return;

  int pIdx = depth[i+w*j];
  if (pIdx < 0)
    return;
  Primitive &p = dev_primitives[pIdx];
  Fragment &f = dev_fragments[i+w*j];

  glm::vec3 tri[3];
  tri[0] = glm::vec3(p.v[0].pos);
  tri[1] = glm::vec3(p.v[1].pos);
  tri[2] = glm::vec3(p.v[2].pos);
  glm::vec2 coord(i,j);
  glm::vec3 bary = calculateBarycentricCoordinate(tri, coord);
  f.z = FloatUnflip(depth[i+w*j]);

  // lambert
  f.eyePos = glm::normalize(bary.x*p.v[0].eyePos
                          + bary.y*p.v[1].eyePos
                          + bary.z*p.v[2].eyePos);
  f.eyeNor = glm::normalize(bary.x*p.v[0].eyeNor
                          + bary.y*p.v[1].eyeNor
                          + bary.z*p.v[2].eyeNor);
  glm::vec3 lambert = glm::clamp(-glm::vec3(glm::dot(f.eyePos, f.eyeNor)), 0.0f, 1.0f);

  glm::vec3 texBary = bary / glm::vec3(p.v[0].pos[3], p.v[1].pos[3], p.v[2].pos[3]);
  glm::vec2 st0 = texBary[0]*p.v[0].texcoord0
        + texBary[1]*p.v[1].texcoord0
        + texBary[2]*p.v[2].texcoord0;
  st0 /= texBary[0] + texBary[1] + texBary[2];

  if (p.v[0].dev_diffuseTex) {
    float4 rgba = tex2D<float4>(p.v[0].dev_diffuseTexObj, st0.x, st0.y);
    lambert *= glm::vec3(rgba.x,rgba.y,rgba.z);
  }
  framebuffer[i + w*j] = lambert;
}

__device__ static inline float smoothstep(float a, float b, float x) {
  x = (x-a)/(b-a);
  x = (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
  return x*x*(3.0f - 2.0f*x);
}

__global__ void ssaoPass(int w, int h, Fragment *dev_fragments, const glm::mat4 P,
                          int ssaoTexSize, int ssaoTexObj,
                          int ssaoKernSize, const glm::vec3 *ssaoKern,
                          float ssaoRadius, unsigned long long *depth, glm::vec3 *framebuffer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= w || j >= h)
    return;
  Fragment &f = dev_fragments[i + w*j];

  float4 rVec4 = tex2D<float4>(ssaoTexObj, float(i)/ssaoTexSize, float(j)/ssaoTexSize);
  glm::vec3 rVec(rVec4.x, rVec4.y, 0.0);
  glm::vec3 tVec = glm::normalize(rVec - f.eyeNor * glm::dot(rVec, f.eyeNor));
  glm::vec3 bVec = glm::cross(f.eyeNor, tVec);
  glm::mat3 TBN(tVec, bVec, f.eyeNor);

  float ssao = 0.0f;
  for (int k = 0; k < ssaoKernSize*ssaoKernSize; k++) {
    glm::vec4 samp = P*glm::vec4(ssaoRadius*TBN*ssaoKern[k] + f.eyePos, 1.0f);
    int si = 0.5f * w * (samp.x + 1.0f);
    int sj = 0.5f * h * (1.0f - samp.y);
    float z = dev_fragments[si + w*sj].z;
    if (z > f.z)
      ssao += smoothstep(0.0, 1.0, ssaoRadius / fabs(f.z - z));
  }
  ssao = 1.0 - ssao/(ssaoKernSize*ssaoKernSize);
  framebuffer[i+w*j] = glm::vec3(ssao);
}

__global__ void ssaoPassShared(int w, int h, Fragment *dev_fragments, const glm::mat4 P,
                          int ssaoTexSize, int ssaoTexObj,
                          int ssaoKernSize, const glm::vec3 *ssaoKern, float ssaoRadius) {
  extern __shared__ Fragment sFrag[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= w || j >= h)
    return;

  // load the fragment into shared memory
  int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
  sFrag[tIdx] = dev_fragments[i + w*j];
  Fragment &f = sFrag[tIdx];

  float4 rVec4 = tex2D<float4>(ssaoTexObj, float(i)/ssaoTexSize, float(j)/ssaoTexSize);
  glm::vec3 rVec(rVec4.x, rVec4.y, 0.0);
  glm::vec3 tVec = glm::normalize(rVec - f.eyeNor * glm::dot(rVec, f.eyeNor));
  glm::vec3 bVec = glm::cross(f.eyeNor, tVec);
  glm::mat3 TBN(tVec, bVec, f.eyeNor);

  f.ssao = 0.0;
  for (int k = 0; k < ssaoKernSize*ssaoKernSize; k++) {
    glm::vec4 samp = P*glm::vec4(ssaoRadius*TBN*ssaoKern[k] + f.eyePos, 1.0f);
    int si = glm::floor(0.5f * w * (samp.x + 1.0f));
    int sj = glm::floor(0.5f * h * (1.0f - samp.y));
    if (si >= 0 && sj >= 0 && si < w && sj < h) {
      float z = dev_fragments[si + w*sj].z;
      if (z > f.z)
        f.ssao += smoothstep(0.0, 1.0, ssaoRadius / fabs(f.z - z));
    }
  }
  dev_fragments[i+w*j].ssao = 1.0 - f.ssao/(ssaoKernSize*ssaoKernSize);
}

__global__ void ssaoBlur(int w, int h, Fragment *dev_fragments, int ssaoTexSize, glm::vec3 *framebuffer) {
  extern __shared__ Fragment sFrag[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= w || j >= h)
    return;

  // load the fragment into shared memory
  int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
  int idx = i + w*j;
  sFrag[tIdx] = dev_fragments[idx];

  float ssao = 0.0f;
  int n = 0;
  for (int sj = j-ssaoTexSize/2; sj < j+ssaoTexSize/2; sj++) {
  for (int si = i-ssaoTexSize/2; si < i+ssaoTexSize/2; si++) {
    if (si >= 0 && sj >= 0 && si < w && sj < h)
      ssao += dev_fragments[si + w*sj].ssao;
    else
      continue;
    n++;
  }}
  ssao /= n;
  framebuffer[idx] *= ssao;
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, const glm::mat4 &P) {
  int sideLength2d = 16;
  dim3 blockSize2d(sideLength2d, sideLength2d);
  dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
  (height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
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

      _vertexTransformAndAssembly <<< numBlocksForVertices, numThreadsPerBlock >>>(p->numVertices, *p, MVP, MV, MV_normal, width, height);
      checkCUDAError("Vertex Processing");
      cudaDeviceSynchronize();
      _primitiveAssembly <<< numBlocksForIndices, numThreadsPerBlock >>>
        (p->numIndices,
        curPrimitiveBeginId,
        dev_primitives,
        *p);
      checkCUDAError("Primitive Assembly");

      curPrimitiveBeginId += p->numPrimitives;
    }
  }
  checkCUDAError("Vertex Processing and Primitive Assembly");

	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  cudaMemset(dev_depth, 0xFF, width * height * sizeof(unsigned long long));
  cudaMemset(dev_framebuffer, 0, width*height*sizeof(glm::vec3));

  dim3 blockDim1d(1024);
  dim3 blockCnt1d((totalNumPrimitives + blockDim1d.x - 1)/blockDim1d.x);

  depthPass<<<blockCnt1d,blockDim1d>>>(totalNumPrimitives, dev_primitives, width, height, dev_depth);
  checkCUDAError("fragDepthFind");

  _fragRasterize<<<blockCount2d,blockSize2d>>>(totalNumPrimitives, dev_primitives, dev_fragmentBuffer, width, height, dev_depth, dev_framebuffer);
  checkCUDAError("fragRasterize");

  //cudaMemset(dev_framebuffer, 0, width*height*sizeof(glm::vec3));
  int smSize = sideLength2d*sideLength2d*sizeof(Fragment);
  ssaoPassShared<<<blockCount2d,blockSize2d,smSize>>>(width, height, dev_fragmentBuffer, P, 4, dev_ssaoTexObj, 8, dev_ssaoKernel, 5.0);
  checkCUDAError("fragRasterize");

  ssaoBlur<<<blockCount2d,blockSize2d,smSize>>>(width, height, dev_fragmentBuffer, 4, dev_framebuffer);
  checkCUDAError("fragRasterize");

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
			cudaFreeArray(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);
		}
	}

	////////////

  cudaFree(dev_primitives);
  dev_primitives = NULL;

  cudaFree(dev_tiles_min);
  cudaFree(dev_tiles_max);
  dev_tiles_min = dev_tiles_max = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

  cudaFree(dev_framebuffer);
  dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = nullptr;

  checkCUDAError("rasterize Free");
}
