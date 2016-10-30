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

#define LAMBERT_SHADING 1
#define TILE_BASED_RASTERIZATION 1
#define SSAA_LEVEL 2

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
		glm::vec2 texcoord0;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
		TextureData* diffuse_tex = nullptr;
		int diffuse_tex_width, diffuse_tex_height;
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* diffuse_tex;
		int diffuse_tex_width;
		int diffuse_tex_height;
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
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static int output_width = 0;
static int output_height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

#if TILE_BASED_RASTERIZATION
// FIXME: tile size is hard to manage

static int tile_w_count = 0;
static int tile_h_count = 0;

const int tile_width = 16;
const int tile_height = 16;

const int max_tile_prim_count = 64;
static Primitive * dev_tile_primitives = nullptr;
static int * dev_tile_prim_counts = nullptr;
#endif
static int * dev_frag_mutex = nullptr;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, int render_w, int render_h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		glm::vec3 color (0.f);
		for (int render_x = SSAA_LEVEL * x; render_x < SSAA_LEVEL * x + SSAA_LEVEL; render_x++)
		{
			for (int render_y = SSAA_LEVEL * y; render_y < SSAA_LEVEL * y + SSAA_LEVEL; render_y++)
			{
				auto fbuffer_index = render_x + render_y * w * SSAA_LEVEL;
				color.x = color.x + glm::clamp(image[fbuffer_index].x, 0.0f, 1.0f) * 255.0;
				color.y = color.y + glm::clamp(image[fbuffer_index].y, 0.0f, 1.0f) * 255.0;
				color.z = color.z + glm::clamp(image[fbuffer_index].z, 0.0f, 1.0f) * 255.0;
			}
		}

		color /= (SSAA_LEVEL * SSAA_LEVEL);

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
void render(int w, int h, const Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h)
	{
		// DONE: add your fragment shader code here
		auto frag = fragmentBuffer[index]; // copy to local mem
		glm::vec3 color;

		// Base Color
		if (!frag.diffuse_tex)
		{
			color = frag.color;
		}
		else
		{
			int tx = static_cast<int>(frag.texcoord0.x * frag.diffuse_tex_width) % frag.diffuse_tex_width;
			if (tx < 0) { tx += frag.diffuse_tex_width; }
			int ty = static_cast<int>(frag.texcoord0.y * frag.diffuse_tex_height) % frag.diffuse_tex_height;
			if (ty < 0) { ty += frag.diffuse_tex_height; }
			int pixel_index = 3 * (tx + ty * frag.diffuse_tex_width);
			
			color = glm::vec3(
				frag.diffuse_tex[pixel_index] / 255.f,
				frag.diffuse_tex[pixel_index + 1] / 255.f,
				frag.diffuse_tex[pixel_index + 2] / 255.f
			);
		}

		// Lighting
#if LAMBERT_SHADING
		const auto light_dir = glm::vec3(1.f, 1.f, 1.f);
		color *= fmaxf(0.f, glm::dot(light_dir, frag.eyeNor));
#endif
		// output
		framebuffer[index] = color;
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	output_width = w;
	output_height = h;
    width = w * SSAA_LEVEL;
    height = h * SSAA_LEVEL;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

#if TILE_BASED_RASTERIZATION
	tile_w_count = (width - 1) / tile_width + 1;
	tile_h_count = (height - 1) / tile_height + 1;
	if (dev_tile_primitives) { cudaFree(dev_tile_primitives); }
	cudaMalloc(&dev_tile_primitives
		, tile_w_count * tile_h_count * max_tile_prim_count * sizeof(dev_tile_primitives[0]));
	if (dev_tile_prim_counts) { cudaFree(dev_tile_prim_counts); }
	cudaMalloc(&dev_tile_prim_counts
		, tile_w_count * tile_h_count * sizeof(dev_tile_prim_counts[0]));
	cudaMemset(dev_tile_prim_counts, 0
		, tile_w_count * tile_h_count * sizeof(dev_tile_prim_counts[0]));
#endif
	if (dev_frag_mutex) { cudaFree(dev_frag_mutex); }
	cudaMalloc(&dev_frag_mutex, width * height * sizeof(dev_frag_mutex[0]));
	cudaMemset(dev_frag_mutex, 0, width * height * sizeof(dev_frag_mutex[0]));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepthAndMutex(int w, int h, int * depth, int* mutex)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
		mutex[index] = 0;
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
	if (vid >= numVertices) { return; }

	// DONE: Apply vertex transformation here
	// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
	auto pos = MVP * glm::vec4(primitive.dev_position[vid], 1.f);
	// Then divide the pos by its w element to transform into NDC space
	pos /= pos.w;
	// Finally transform x and y to viewport space
	pos.x = (- pos.x * 0.5f + 0.5f) * width;
	pos.y = (- pos.y * 0.5f + 0.5f) * height;

	// DONE: Apply vertex assembly here
	// Assemble all attribute arraies into the primitive array
	auto eye_pos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.f));
	auto eye_normal = glm::normalize(MV_normal * primitive.dev_normal[vid]);
	VertexAttributeTexcoord tex_coord(0.f);
	if (primitive.dev_texcoord0)
	{
		tex_coord = primitive.dev_texcoord0[vid];
	}
	auto tex_diffuse = primitive.dev_diffuseTex;

	auto& v_out = primitive.dev_verticesOut[vid];
	v_out.pos = pos;
	v_out.eyePos = eye_pos;
	v_out.eyeNor = eye_normal;
	v_out.texcoord0 = tex_coord;
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// DONE: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) 
		{
			pid = iid / (int)primitive.primitiveType;
			auto& out_primitive = dev_primitives[pid + curPrimitiveBeginId];
			auto v_index = iid % (int)primitive.primitiveType;
			out_primitive.v[v_index] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
			out_primitive.primitiveType = primitive.primitiveType;

			if (v_index == 0)
			{
				out_primitive.diffuse_tex = primitive.dev_diffuseTex;
				out_primitive.diffuse_tex_width = primitive.diffuseTexWidth;
				out_primitive.diffuse_tex_height = primitive.diffuseTexHeight;
			}
		}
		// TODO: other primitive types (point, line)
	}
	
}

__device__ void rasterizeTriangleFrag(
	const Primitive& primitive
	, const glm::vec3 tri[]
	, int x
	, int y
	, int width
	, int height
	, Fragment * frag_buffer
	, int * depth_buffer
	, int * mutex_buffer
)
{
	if (!(x >= 0 && x < width && y >= 0 && y < height))
	{
		return;
	}

	auto bary_coord = calculateBarycentricCoordinate(tri, glm::vec2(x, y));

	if (!isBarycentricCoordInBounds(bary_coord))
	{
		return;
	}

	auto frag_index = x + y * width;

	int depth = -getZAtCoordinate(bary_coord, tri) * INT_MAX;

	//// lock mutex
	//while (true)
	//{
	//	if (atomicCAS(mutex_buffer + frag_index, 0, 1) == 0)
	//	{
	//		// mutex locked

	//		if (depth < depth_buffer[frag_index])
	//		{
	//			depth_buffer[frag_index] = depth;
	//			frag_buffer[frag_index].color = glm::vec3(1.f);
	//			frag_buffer[frag_index].diffuse_tex = primitive.diffuse_tex;
	//			frag_buffer[frag_index].diffuse_tex_height = primitive.diffuse_tex_height;
	//			frag_buffer[frag_index].diffuse_tex_width = primitive.diffuse_tex_width;

	//			frag_buffer[frag_index].eyePos = baryInterpolate(bary_coord, primitive.v[0].eyePos, primitive.v[1].eyePos, primitive.v[2].eyePos);
	//			frag_buffer[frag_index].eyeNor = baryInterpolate(bary_coord, primitive.v[0].eyeNor, primitive.v[1].eyeNor, primitive.v[2].eyeNor);
	//			frag_buffer[frag_index].texcoord0 = baryInterpolate(bary_coord, primitive.v[0].texcoord0, primitive.v[1].texcoord0, primitive.v[2].texcoord0);
	//		}

	//		// unlock mutex
	//		atomicExch(mutex_buffer + frag_index, 0);
	//		break;
	//	}
	//}
	//atomicExch(mutex_buffer + frag_index, 0);
#if TILE_BASED_RASTERIZATION
	if (depth > depth_buffer[frag_index]) { return; }
	depth_buffer[frag_index] = depth;
#else
	atomicMin(&depth_buffer[frag_index], depth);
	if (depth != depth_buffer[frag_index])
	{
		return;
	}
#endif

	frag_buffer[frag_index].color = glm::vec3(0.5f);
	frag_buffer[frag_index].diffuse_tex = primitive.diffuse_tex;
	frag_buffer[frag_index].diffuse_tex_height = primitive.diffuse_tex_height;
	frag_buffer[frag_index].diffuse_tex_width = primitive.diffuse_tex_width;
	//interpolate
	frag_buffer[frag_index].eyePos = baryInterpolate(bary_coord, primitive.v[0].eyePos, primitive.v[1].eyePos, primitive.v[2].eyePos);
	frag_buffer[frag_index].eyeNor = baryInterpolate(bary_coord, primitive.v[0].eyeNor, primitive.v[1].eyeNor, primitive.v[2].eyeNor);
	frag_buffer[frag_index].texcoord0 = baryInterpolate(bary_coord, primitive.v[0].texcoord0, primitive.v[1].texcoord0, primitive.v[2].texcoord0);

}

#if TILE_BASED_RASTERIZATION

//static int tile_w_count = 0;
//static int tile_h_count = 0;
//
//const int tile_width = 64;
//const int tile_height = 64;
//
//const int max_tile_prim_count = 32;
//static Primitive * dev_tile_primitives = nullptr;
//static int * dev_tile_prim_counts = nullptr;
__global__ void addPrimitivesToTiles(
	int num_primitives
	, const Primitive* primitives
	, int width
	, int height
	, int tile_width
	, int tile_height
	, int tile_prim_count_limit
	, Primitive* tile_primitives
	, int * tile_prim_counts
)
{
	// index id
	auto pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid >= num_primitives) { return; }
	// copy primitive data to local memory
	auto primitive = primitives[pid];

	if (primitive.primitiveType == PrimitiveType::Triangle)
	{
		glm::vec2 aabb_min = {
			fmaxf(fminf(fminf(primitive.v[0].pos[0],primitive.v[1].pos[0]) , primitive.v[2].pos[0]) , 0)
			, fmaxf(fminf(fminf(primitive.v[0].pos[1],primitive.v[1].pos[1]) , primitive.v[2].pos[1]) , 0)
		};
		glm::vec2 aabb_max = {
			fminf(fmaxf(fmaxf(primitive.v[0].pos[0],primitive.v[1].pos[0]) , primitive.v[2].pos[0]) , width - 1)
			, fminf(fmaxf(fmaxf(primitive.v[0].pos[1],primitive.v[1].pos[1]) , primitive.v[2].pos[1]) , height - 1)
		};

		auto min_x_tile = static_cast<int>(aabb_min.x) / tile_width;
		auto min_y_tile = static_cast<int>(aabb_min.y) / tile_height;
		auto max_x_tile = static_cast<int>(aabb_max.x) / tile_width;
		auto max_y_tile = static_cast<int>(aabb_max.y) / tile_height;
		auto tile_x_count = (width - 1) / tile_width + 1;
		for (int tx = min_x_tile; tx <= max_x_tile; tx++)
		{
			for (int ty = min_y_tile; ty <= max_y_tile; ty++)
			{
				auto tile_id = tx + ty * tile_x_count;
				auto prim_slot = atomicAdd(tile_prim_counts + tile_id, 1);
				if (prim_slot >= tile_prim_count_limit)
				{
					continue;
					// TODO: make tile able to contain more primitives somehow
				}
				tile_primitives[tile_id * tile_prim_count_limit + prim_slot] = primitive;
			}
		}
	}
}

__global__ void kernRasterizeTiles(
	int tile_x_count
	, int tile_y_count
	, int tile_width
	, int tile_height
	, int width
	, int height
	, Primitive* tile_primitives
	, int * tile_prim_counts
	, int tile_prim_count_limit
	, Fragment * frag_buffer
	, int * depth_buffer
)
{
	int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (!(tx >= 0 && tx < tile_x_count && ty >= 0 && ty < tile_y_count))
	{
		return;
	}
	int index = tx + (ty * tile_x_count);

	int x_begin = tx * tile_width;
	int x_end = glm::min(x_begin + tile_width, width);
	int y_begin = ty * tile_height;
	int y_end = glm::min(y_begin + tile_height, height);

	auto prim_count = glm::min(tile_prim_counts[index], tile_prim_count_limit);
	for (int y = y_begin; y < y_end; y++)
	{
		for (int x = x_begin; x < x_end; x++)
		{
			for (int i = 0; i < prim_count; i++)
			{
				auto& prim = tile_primitives[index * tile_prim_count_limit + i];
				glm::vec3 tri_pos[3] = { glm::vec3(prim.v[0].pos)
					, glm::vec3(prim.v[1].pos)
					, glm::vec3(prim.v[2].pos)
				};
				rasterizeTriangleFrag(prim, tri_pos, x, y, width, height, frag_buffer, depth_buffer, nullptr);
			}
		}
	}

}

#else

__global__ void kernRasterizePrimitives(
	int num_primitives
	, const Primitive* primitives
	, int width
	, int height
	, Fragment * frag_buffer
	, int * depth_buffer
	, int * mutex_buffer
	)
{
	// index id
	auto pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid >= num_primitives) { return; }
	// copy primitive data to local memory
	auto primitive = primitives[pid]; 

	if (primitive.primitiveType == PrimitiveType::Triangle)
	{
		glm::vec2 aabb_min = { 
			fmaxf(fminf(fminf( primitive.v[0].pos[0],primitive.v[1].pos[0]) , primitive.v[2].pos[0]) , 0)
			, fmaxf(fminf(fminf(primitive.v[0].pos[1],primitive.v[1].pos[1]) , primitive.v[2].pos[1]) , 0)
			};
		glm::vec2 aabb_max = {
			fminf(fmaxf(fmaxf(primitive.v[0].pos[0],primitive.v[1].pos[0]) , primitive.v[2].pos[0]) , width - 1)
			, fminf(fmaxf(fmaxf(primitive.v[0].pos[1],primitive.v[1].pos[1]) , primitive.v[2].pos[1]) , height - 1)
			};

		// TODO: CUDA Dynamic Parallelism?
		glm::vec3 tri_pos[3] = { glm::vec3(primitive.v[0].pos) 
			, glm::vec3(primitive.v[1].pos) 
			, glm::vec3(primitive.v[2].pos) 
			};
		for (int x = aabb_min.x; x <= static_cast<int>(aabb_max.x); x++)
		{
			for (int y = aabb_min.y; y <= static_cast<int>(aabb_max.y); y++)
			{
				rasterizeTriangleFrag(primitive, tri_pos, x, y, width, height, frag_buffer, depth_buffer, mutex_buffer);
			}
		}
	}
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
	initDepthAndMutex << <blockCount2d, blockSize2d >> >(width, height, dev_depth, dev_frag_mutex);
	

	// TODO: rasterize
	{
#if TILE_BASED_RASTERIZATION 
		cudaMemset(dev_tile_prim_counts, 0
			, tile_w_count * tile_h_count * sizeof(dev_tile_prim_counts[0]));

		dim3 numThreadsPerBlock(128);
		dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
		addPrimitivesToTiles << <numBlocksForPrimitives, numThreadsPerBlock >> > (
			totalNumPrimitives
			, dev_primitives
			, width
			, height
			, tile_width
			, tile_height
			, max_tile_prim_count
			, dev_tile_primitives
			, dev_tile_prim_counts
		);
		checkCUDAError("addPrimitivesToTiles");

		dim3 tile_blockSize2d(8, 8);
		dim3 tile_blockCount2d((tile_w_count - 1) / tile_blockSize2d.x + 1,
			(tile_h_count - 1) / tile_blockSize2d.y + 1);
		kernRasterizeTiles << <tile_blockCount2d, tile_blockSize2d >> >(
			tile_w_count
			, tile_h_count
			, tile_width
			, tile_height
			, width
			, height
			, dev_tile_primitives
			, dev_tile_prim_counts
			, max_tile_prim_count
			, dev_fragmentBuffer
			, dev_depth
		);
		checkCUDAError("kernRasterizeTiles");
#else
		dim3 numThreadsPerBlock(128);
		dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
		kernRasterizePrimitives << <numBlocksForPrimitives, numThreadsPerBlock >> >(totalNumPrimitives, dev_primitives, width, height, dev_fragmentBuffer, dev_depth, dev_frag_mutex);
		checkCUDAError("Rasterization");
#endif
	}

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, output_width, output_height, width, height, dev_framebuffer);
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

#if TILE_BASED_RASTERIZATION
	cudaFree(dev_tile_primitives);
	dev_tile_primitives = nullptr;

	cudaFree(dev_tile_prim_counts);
	dev_tile_prim_counts = nullptr;
#endif
	cudaFree(dev_frag_mutex);
	dev_frag_mutex = nullptr;

    checkCUDAError("rasterize Free");
}
