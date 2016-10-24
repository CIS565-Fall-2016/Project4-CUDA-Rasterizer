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

#define PERSPECTIVE_CORRECT
#define BILIN_INTERP
#define CONSTANT_MEM
#define SEPARATE_INTERP
#define OPTIM_BARY // https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
//#define RASTERIZE_BY_PIXEL
#define TILED_RENDERING
#define TILE_SIZE 32

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
		TextureData* dev_diffuseTex;
    int texWidth, texHeight;
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
    int texWidth, texHeight;

    Primitive* prim;
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
    int texWidth, texHeight;
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
static Primitive* *dev_primitiveBuffer = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

#ifdef CONSTANT_MEM
__constant__ float c_MVP[sizeof(glm::mat4) / sizeof(float)];
__constant__ float c_MV[sizeof(glm::mat4) / sizeof(float)];
__constant__ float c_MV_normal[sizeof(glm::mat3) / sizeof(float)];
__constant__ int c_width[1];
__constant__ int c_height[1];
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

// get RGB value at index as 01 floats
__host__ __device__ static
glm::vec3 getRGB(TextureData* tex, int width, int x, int y)
{
  int idx = 3 * (x + y * width);
  return glm::vec3(
    (float)(tex[idx]) / 255.f,
    (float)(tex[idx + 1]) / 255.f,
    (float)(tex[idx + 2]) / 255.f
  );
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
#ifdef CONSTANT_MEM
void render(Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
#else
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
#endif
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
#ifdef CONSTANT_MEM
    int index = x + (y * c_width[0]);
#else
    int index = x + (y * w);
#endif
      
    const glm::vec3 lightVec = glm::normalize(glm::vec3(1, 3, 5));
    int twidth = fragmentBuffer[index].texWidth;
    int theight = fragmentBuffer[index].texHeight;

    glm::vec3 col;
#ifdef CONSTANT_MEM
    if (x < c_width[0] && y < c_height[0] && x > 0 && y > 0) {
#else
    if (x < w && y < h && x > 0 && y > 0) {
#endif
      // TODO: add your fragment shader code here
      if (fragmentBuffer[index].eyePos.z != 0) {
        if (fragmentBuffer[index].dev_diffuseTex != NULL) {

          TextureData* tex = fragmentBuffer[index].dev_diffuseTex;
#ifdef BILIN_INTERP
          float tx = fragmentBuffer[index].texcoord0.x * twidth;
          float ty = fragmentBuffer[index].texcoord0.y * theight;
          int x_low = tx;
          int x_high = tx + 1;
          int y_low = ty;
          int y_high = ty + 1;
          int x_clamp = __min(x_high, twidth - 1);
          int y_clamp = __min(y_high, theight - 1);

          col =
            (y_high - ty) * (
            (x_high - tx) * getRGB(tex, twidth, x_low, y_low) +
            (tx - x_low) * getRGB(tex, twidth, x_clamp, y_low)
            ) +
            (ty - y_low) * (
            (x_high - tx) * getRGB(tex, twidth, x_low, y_clamp) +
            (tx - x_low) * getRGB(tex, twidth, x_clamp, y_clamp)
            );
#else
          int tx = fragmentBuffer[index].texcoord0.x * twidth;
          int ty = fragmentBuffer[index].texcoord0.y * theight;
          col = getRGB(tex, twidth, tx, ty);
#endif
        }
        else {
          col = fragmentBuffer[index].eyeNor;
        }
      }
      framebuffer[index] = glm::dot(lightVec, fragmentBuffer[index].eyeNor) * col;

    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;

#ifdef CONSTANT_MEM
    cudaMemcpyToSymbol(c_width, &width, sizeof(int));
    cudaMemcpyToSymbol(c_height, &height, sizeof(int));
#endif

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

  cudaFree(dev_primitiveBuffer);
  cudaMalloc(&dev_primitiveBuffer, width * height * sizeof(Primitive*));
  cudaMemset(dev_primitiveBuffer, 0, width * height * sizeof(Primitive*));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

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
					VertexIndex* dev_indices;
					VertexAttributePosition* dev_position;
					VertexAttributeNormal* dev_normal;
					VertexAttributeTexcoord* dev_texcoord0;
          int texWidth;
          int texHeight;

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
                  texHeight = image.width;

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


#ifdef CONSTANT_MEM
__global__ 
void _vertexTransformAndAssembly(
int numVertices, 
PrimitiveDevBufPointers primitive) {

  // vertex id
  int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vid < numVertices) {

    // TODO: Apply vertex transformation here
    // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    // Then divide the pos by its w element to transform into NDC space
    // Finally transform x and y to viewport space
    glm::mat4& MVP = *(glm::mat4*)c_MVP;
    glm::mat4& MV = *(glm::mat4*)c_MV;
    glm::mat3& MV_normal = *(glm::mat3*)c_MV_normal;

    glm::vec4 screenV = MVP * glm::vec4(primitive.dev_position[vid], 1.f);
    glm::vec4 pixelV = screenV / screenV.w;
    pixelV.x = (1.f - pixelV.x) / 2.f * c_width[0];
    pixelV.y = (1.f - pixelV.y)/2.f * c_height[0];
    primitive.dev_verticesOut[vid].pos = pixelV;

    // TODO: Apply vertex assembly here
    // Assemble all attribute arraies into the primitive array
    primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.f));
    primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
    primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
  }
}

#else
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
    glm::vec4 screenV = MVP * glm::vec4(primitive.dev_position[vid], 1.f);
    glm::vec4 pixelV = screenV / screenV.w;
    pixelV.x = (1.f - pixelV.x) / 2.f * width;
    pixelV.y = (1.f - pixelV.y)/2.f * height;
    primitive.dev_verticesOut[vid].pos = pixelV;

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
    primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.f));
    primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
    primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
	}
}
#endif



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
      // each .v is of length primitiveMode. if tries, the first three indices are the three items here
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
      dev_primitives[pid + curPrimitiveBeginId].dev_diffuseTex = primitive.dev_diffuseTex;
      dev_primitives[pid + curPrimitiveBeginId].texWidth = primitive.texWidth;
      dev_primitives[pid + curPrimitiveBeginId].texHeight = primitive.texHeight;
		}


		// TODO: other primitive types (point, line)
	}
	
}

__global__
void _rasterizeByPixel(int numPrimitives, Primitive* primitives, Fragment* fragments, int width, int height) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int idx = x + y * width;
  
  int pid;
  float closest = -1;
  int pidx = -1;
  for (pid = 0; pid < numPrimitives; ++pid) {
    glm::vec3 triangle[3];
    triangle[0] = glm::vec3(primitives[pid].v[0].pos);
    triangle[1] = glm::vec3(primitives[pid].v[1].pos);
    triangle[2] = glm::vec3(primitives[pid].v[2].pos);
    glm::vec3 bCoord = calculateBarycentricCoordinate(triangle, glm::vec2(x, y));
    if (isBarycentricCoordInBounds(bCoord)) {
      float z = -getZAtCoordinate(bCoord, triangle);
      if (z < closest || closest == -1) {
        closest = z;
        pidx = pid;
      }
    }
  }
  if (pidx >= 0) fragments[idx].prim = primitives + pidx;
}

__global__
void _rasterizePrims(int numPrimitives, Primitive* primitives, int* depths, Primitive** prims, int width, int height) {
    // primitive id
    int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pid < numPrimitives) {
      Primitive &prim = primitives[pid];

#ifdef OPTIM_BARY
      glm::vec2 v0 = glm::vec2(prim.v[0].pos);
      glm::vec2 v1 = glm::vec2(prim.v[1].pos);
      glm::vec2 v2 = glm::vec2(prim.v[2].pos);
      AABBScreen box = getAABBForTriangleScreen(v0, v1, v2);
#else
      glm::vec3 triangle[3] = { glm::vec3(prim.v[0].pos),
                                glm::vec3(prim.v[1].pos),
                                glm::vec3(prim.v[2].pos) };
      AABB box = getAABBForTriangle(triangle);
#endif

      // clip bbox to screen
      box.min.x = __max(box.min.x, 0);
      box.max.x = __min(box.max.x, width);
      box.min.y = __max(box.min.y, 0);
      box.max.y = __min(box.max.y, height);

#ifdef OPTIM_BARY
      float A01 = v0.y - v1.y, B01 = v1.x - v0.x;
      float A12 = v1.y - v2.y, B12 = v2.x - v1.x;
      float A20 = v2.y - v0.y, B20 = v0.x - v2.x;

      glm::vec2 p = box.min;

      float w0_row = calculateSignedParallelogramArea(v1, p, v2);
      float w1_row = calculateSignedParallelogramArea(v2, p, v0);
      float w2_row = calculateSignedParallelogramArea(v0, p, v1);

      for (p.y = box.min.y; p.y <= box.max.y; p.y++) {
        float w0 = w0_row;
        float w1 = w1_row;
        float w2 = w2_row;

        for (p.x = box.min.x; p.x <= box.max.x; p.x++) {
          if (w0 >= 0 && w1 >=0 && w2 >= 0) {
            int pix_idx = p.x + p.y * width;
            int intz = INT_MAX * (w0 * prim.v[0].pos.z + w1 * prim.v[1].pos.z + w2 * prim.v[2].pos.z) / (w0 + w1 + w2);
            atomicMin(&depths[pix_idx], intz);
            if (depths[pix_idx] == intz) {
              prims[pix_idx] = &prim;
              //fragments[pix_idx].prim = &prim;
            }
          }

          w0 += A12;
          w1 += A20;
          w2 += A01;
        }

        w0_row += B12;
        w1_row += B20;
        w2_row += B01;
      }

#else
      for (int x = box.min.x; x < box.max.x; ++x) {
        for (int y = box.min.y; y < box.max.y; ++y) {
          int pix_idx = x + y * width;
          glm::vec3 bCoord = calculateBarycentricCoordinate(triangle, glm::vec2(x, y));
          if (isBarycentricCoordInBounds(bCoord)) {
            int intz = INT_MAX * -getZAtCoordinate(bCoord, triangle);
            atomicMin(&depths[pix_idx], intz);
            if (depths[pix_idx] == intz) {
              fragments[pix_idx].prim = &prim;
            }
          }
        }
      }
#endif
    }
}

__global__
void _interpolateAttributes(Fragment* fragments, Primitive** prims, int width, int height) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + y * width;

    if (x < width && y < height) {
      if (prims[index] == NULL) return;
      Fragment& frag = fragments[index];
      Primitive& prim = *prims[index];

      glm::vec3 triangle[3] = { glm::vec3(prim.v[0].pos), glm::vec3(prim.v[1].pos), glm::vec3(prim.v[2].pos) };
#ifdef PERSPECTIVE_CORRECT
      float eye_zs[3] = { prim.v[0].eyePos.z, prim.v[1].eyePos.z, prim.v[2].eyePos.z };
#endif
      glm::vec3 bCoord = calculateBarycentricCoordinate(triangle, glm::vec2(x, y));
#ifdef PERSPECTIVE_CORRECT
      float correct_z = getCorrectZAtCoordinate(bCoord, eye_zs);
#endif

#ifdef PERSPECTIVE_CORRECT
      frag.eyeNor = correct_z * (
        bCoord.x * prim.v[0].eyeNor / eye_zs[0] +
        bCoord.y * prim.v[1].eyeNor / eye_zs[1] +
        bCoord.z * prim.v[2].eyeNor / eye_zs[2]
        );
      frag.eyePos = correct_z * (
        bCoord.x * prim.v[0].eyePos / eye_zs[0] +
        bCoord.y * prim.v[1].eyePos / eye_zs[1] +
        bCoord.z * prim.v[2].eyePos / eye_zs[2]
        );
      frag.texcoord0 = correct_z * (
        bCoord.x * prim.v[0].texcoord0 / eye_zs[0] +
        bCoord.y * prim.v[1].texcoord0 / eye_zs[1] +
        bCoord.z * prim.v[2].texcoord0 / eye_zs[2]
        );
#else
      frag.eyeNor =
        bCoord.x * prim.v[0].eyeNor +
        bCoord.y * prim.v[1].eyeNor +
        bCoord.z * prim.v[2].eyeNor;
      frag.eyePos =
        bCoord.x * prim.v[0].eyePos +
        bCoord.y * prim.v[1].eyePos +
        bCoord.z * prim.v[2].eyePos;
      frag.texcoord0 =
        bCoord.x * prim.v[0].texcoord0 +
        bCoord.y * prim.v[1].texcoord0 +
        bCoord.z * prim.v[2].texcoord0;
#endif
      frag.dev_diffuseTex = prim.dev_diffuseTex;
      frag.texWidth = prim.texWidth;
      frag.texHeight = prim.texHeight;
    }
}

__global__
void _rasterize(int numPrimitives, Primitive* primitives, int* depths, Fragment* fragments, int width, int height) {
  // primitive id
  int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (pid < numPrimitives) {
    Primitive &prim = primitives[pid];
    glm::vec3 triangle[3] = { glm::vec3(prim.v[0].pos),
                              glm::vec3(prim.v[1].pos), 
                              glm::vec3(prim.v[2].pos) };
#ifdef PERSPECTIVE_CORRECT
    float eye_zs[3] = { prim.v[0].eyePos.z, prim.v[1].eyePos.z, prim.v[2].eyePos.z };
#endif

    AABB box = getAABBForTriangle(triangle);
    box.min.x = __max(box.min.x, 0);
    box.max.x = __min(box.max.x, width);
    box.min.y = __max(box.min.y, 0);
    box.max.y = __min(box.max.y, height);

    for (int x = box.min.x; x < box.max.x; ++x) {
      for (int y = box.min.y; y < box.max.y; ++y) {
        int pix_idx = x + y * width;
        glm::vec3 bCoord = calculateBarycentricCoordinate(triangle, glm::vec2(x, y));
        if (isBarycentricCoordInBounds(bCoord)) {

          float z = getZAtCoordinate(bCoord, triangle);
#ifdef PERSPECTIVE_CORRECT
          float correct_z = getCorrectZAtCoordinate(bCoord, eye_zs);
#endif
          int intz = INT_MAX * -z;
          atomicMin(&depths[pix_idx], intz);
          if (depths[pix_idx] == intz) {
            Fragment &frag = fragments[pix_idx];
#ifdef PERSPECTIVE_CORRECT
            frag.eyeNor = correct_z * (
              bCoord.x * prim.v[0].eyeNor / eye_zs[0] +
              bCoord.y * prim.v[1].eyeNor / eye_zs[1] +
              bCoord.z * prim.v[2].eyeNor / eye_zs[2]
              );
            frag.eyePos = correct_z * (
              bCoord.x * prim.v[0].eyePos / eye_zs[0] +
              bCoord.y * prim.v[1].eyePos / eye_zs[1] +
              bCoord.z * prim.v[2].eyePos / eye_zs[2]
              );
            frag.texcoord0 = correct_z * (
              bCoord.x * prim.v[0].texcoord0 / eye_zs[0] +
              bCoord.y * prim.v[1].texcoord0 / eye_zs[1] +
              bCoord.z * prim.v[2].texcoord0 / eye_zs[2]
              );
#else
            frag.eyeNor =
              bCoord.x * prim.v[0].eyeNor +
              bCoord.y * prim.v[1].eyeNor +
              bCoord.z * prim.v[2].eyeNor;
            frag.eyePos =
              bCoord.x * prim.v[0].eyePos +
              bCoord.y * prim.v[1].eyePos +
              bCoord.z * prim.v[2].eyePos;
            frag.texcoord0 =
              bCoord.x * prim.v[0].texcoord0 +
              bCoord.y * prim.v[1].texcoord0 +
              bCoord.z * prim.v[2].texcoord0;
#endif
            frag.dev_diffuseTex = prim.dev_diffuseTex;
            frag.texWidth = prim.texWidth;
            frag.texHeight = prim.texHeight;
          }
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

#ifdef CONSTANT_MEM
  // copy constants
    cudaMemcpyToSymbol(c_MVP, &MVP, sizeof(glm::mat4));
    cudaMemcpyToSymbol(c_MV, &MV, sizeof(glm::mat4));
    cudaMemcpyToSymbol(c_MV_normal, &MV_normal, sizeof(glm::mat3));
#endif

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

#ifdef CONSTANT_MEM
        _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p);
#else
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
#endif
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
  cudaMemset(dev_primitiveBuffer, NULL, width * height * sizeof(Primitive*));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// TODO: rasterize
  dim3 numThreadsPerBlock(128);
  dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
#ifdef SEPARATE_INTERP
#ifdef RASTERIZE_BY_PIXEL
  START_PROFILE(rasterize_pixels)
  _rasterizeByPixel<<<blockCount2d, blockSize2d>>>(totalNumPrimitives, dev_primitives, dev_fragmentBuffer, width, height);
  END_PROFILE(rasterize_pixels)
#else
  START_PROFILE(rasterize_prims)
  _rasterizePrims << <numBlocksForPrimitives, numThreadsPerBlock >> >(totalNumPrimitives, dev_primitives, dev_depth, dev_primitiveBuffer, width, height);
  END_PROFILE(rasterize_prims)
#endif
  START_PROFILE(interpolate)
  _interpolateAttributes << <blockCount2d, blockSize2d >> >(dev_fragmentBuffer, dev_primitiveBuffer, width, height);
  END_PROFILE(interpolate)
#else
  START_PROFILE(rasterize)
    _rasterize << <numBlocksForPrimitives, numThreadsPerBlock >> >(totalNumPrimitives, dev_primitives, dev_depth, dev_fragmentBuffer, width, height);
  END_PROFILE(rasterize)
#endif
  checkCUDAError("Rasterization");

    // Copy depthbuffer colors into framebuffer
#ifdef CONSTANT_MEM
	render << <blockCount2d, blockSize2d >> >(dev_fragmentBuffer, dev_framebuffer);
#else
  render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
#endif
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
