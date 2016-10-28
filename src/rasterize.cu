/**
* @file      rasterize.cu
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
* @date      2012-2016
* @copyright University of Pennsylvania & STUDENT
*/

//Xiang is here

#include "rasterize.h"
#include "common.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#define USETEXTURE 1
#define USELIGHT 1 && USETEXTURE
#define USEBILINFILTER 1 && USETEXTURE
#define USEPERSPECTIVECORRECTION 0 && USETEXTURE
#define USETILE 0
#define USELINES 1  && 1-USETEXTURE
#define USEPOINTS 0 && 1-USETEXTURE
#define SPARSITY 20 //sparsity of point cloud (if on)
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
// MAJOR TODO
/**
* Writes fragment colors to the framebuffer
*/

__device__ __host__
glm::vec3 getTextureVal(int x, int y, int width, int height, TextureData * tex, int texture) {

	if (x < width && y < height&&x >= 0 && y >= 0){
		int id = x + y*width;
		int id0 = texture*id;
		return  glm::vec3(tex[id0], tex[id0 + 1], tex[id0 + 2]) / 255.0f;
	}
	return glm::vec3(0.0f);

}
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	glm::vec3 light = glm::normalize(glm::vec3(1, 2, 3));

	if (x < w && y < h) {
		Fragment & curFrag = fragmentBuffer[index];
		framebuffer[index] = fragmentBuffer[index].color;
#if USELIGHT==1		
		framebuffer[index] *= glm::dot(light, fragmentBuffer[index].eyeNor);
#endif		
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
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

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
	}
	else {
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

void traverseNode(
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

					//Added here 
					int texture = 0;
					///////////////////////////
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

									diffuseTexWidth = image.width;//here changed
									diffuseTexHeight = image.height;
									texture = image.component;

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

						dev_diffuseTex,//here changed
						diffuseTexWidth,
						diffuseTexHeight,
						texture,

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


//MAJOR TODO
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
		 
		
		VertexAttributePosition & curpos = primitive.dev_position[vid];
		VertexOut & out = primitive.dev_verticesOut[vid]; 
		glm::vec4 mvpos = MVP * glm::vec4(curpos, 1.0f);
		glm::vec3 eyepos = glm::vec3(MV * glm::vec4(curpos, 1.0f));
		glm::vec4 ndc = mvpos / mvpos.w;
		ndc.x = (1 - ndc.x)*width / 2;
		ndc.y = (1 - ndc.y)*height / 2;

		out.pos = ndc;
		out.eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		out.eyePos = eyepos;

		out.dev_diffuseTex = primitive.dev_diffuseTex;
		if (primitive.dev_diffuseTex != NULL) {
			out.texcoord0 = primitive.dev_texcoord0[vid];
		}		
		out.texWidth = primitive.texWidth;
		out.texHeight = primitive.texHeight;
		out.texture = primitive.texture;
 

	}
}



static int curPrimitiveBeginId = 0;
//MAJOR TODO Y
__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles
		int pid;// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {//simply copy the attributes for now
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
			dev_primitives[pid + curPrimitiveBeginId].dev_diffuseTex = primitive.dev_diffuseTex;
			dev_primitives[pid + curPrimitiveBeginId].texHeight = primitive.texHeight;
			dev_primitives[pid + curPrimitiveBeginId].texWidth = primitive.texWidth;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType].col = glm::vec3(1.0, 1.0, 0.0);
			//currently default color is red for all
		}
		// TODO: other primitive types (point, line)
	}

}


//MAJOR TODO
/**
* Perform rasterization.
*/
__device__ __host__ int getMax(int a, int b){
	if (a > b){
		return a;
	}
	else{
		return b;
	}
}
__device__ __host__ int getMin(int a, int b){
	if (a < b){
		return a;
	}
	else{
		return b;
	}
}
__global__ void kernTextureMap(int width, int height, Fragment * fragments){
	int idx = threadIdx.x + (blockIdx.x*blockDim.x);
	int idy = threadIdx.y + (blockIdx.y*blockDim.y);
	if (idx < width&&idy < height){
		int index = idx + idy*width;
		Fragment & curFrag = fragments[index];
		if (curFrag.dev_diffuseTex != NULL){
			float tix = 0.5f + curFrag.texcoord0.x * (curFrag.texWidth - 1);
			float tiy = 0.5f + curFrag.texcoord0.y * (curFrag.texHeight - 1);
			int twidth = curFrag.texWidth;
			int theight = curFrag.texHeight;
#if USEBILINFILTER==1
			//reference https://en.wikipedia.org/wiki/Bilinear_filtering
			float u = tix * 1 - 0.5;
			float v = tiy * 1 - 0.5;
			//float u = tix;
			//float v = tiy;
			float x = glm::floor(u);
			float y = glm::floor(v);
			float u_ratio = u - x;
			float v_ratio = v - y;
			float u_opposite = 1.0f - u_ratio;
			float v_opposite = 1.0f - v_ratio;
			glm::vec3 t00 = getTextureVal(x, y, curFrag.texWidth, curFrag.texHeight, curFrag.dev_diffuseTex, curFrag.texture);
			glm::vec3 t01 = getTextureVal(x, y + 1, curFrag.texWidth, curFrag.texHeight, curFrag.dev_diffuseTex, curFrag.texture);
			glm::vec3 t10 = getTextureVal(x + 1, y, curFrag.texWidth, curFrag.texHeight, curFrag.dev_diffuseTex, curFrag.texture);
			glm::vec3 t11 = getTextureVal(x + 1, y + 1, curFrag.texWidth, curFrag.texHeight, curFrag.dev_diffuseTex, curFrag.texture);

			curFrag.color = (t00*u_opposite + t10*u_ratio)*v_opposite + (t01*u_opposite + t11*u_ratio)*v_ratio;
#else
			curFrag.color = getTextureVal(tix, tiy, twidth, theight, curFrag.dev_diffuseTex, curFrag.texture);
#endif
		}
	}

}
__device__ 
glm::vec3 interpoline(glm::vec3 & x, glm::vec3 & y, float len){
	return (1-len) * x + (len) * y;
}
__device__
void devRasterizeLine(glm::vec3& pos, glm::vec3& pos2, glm::vec3 & color,
int width, int height,
Fragment* fragments){ 
	glm::vec3 p;
	int index;
	glm::vec3 d = glm::abs(pos - pos2);
	if (d.x>0 && d.y>0) {

		int len = glm::max(d.x, d.y);

		for (float i = 0; i <= len; ++i) {
			
			p = interpoline(pos, pos2, i / len);
			index = (int)(p.x) + (int)(p.y) * width;
			fragments[index].color = color;
		}
	}
}
__device__
void devRasterizePoints(glm::vec3& pos, glm::vec3& pos2, glm::vec3 & color,
int width, int height,
Fragment* fragments,int stepsize){
	glm::vec3 p;
	int index;
	glm::vec3 d = glm::abs(pos - pos2);
	if (d.x>0 && d.y>0) {

		int len = glm::max(d.x, d.y);

		for (float i = 0; i <= len; i = i + stepsize) {

			p = interpoline(pos, pos2, i / len);
			index = (int)(p.x) + (int)(p.y) * width;
			fragments[index].color = color;
		}
	}
}

__global__ void kernRasterize(int n, Primitive * primitives, int* depths, int width, int height, Fragment* fragments, int randomnum){
	//output: a list of fragments with interpolated attributes
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (index < n){ //   (n too small) this is the crazy bug!!
		Primitive & curPrim = primitives[index];
		VertexOut & vertex0 = curPrim.v[0];
		//if (curPrim.primitiveType == TINYGLTF_MODE_TRIANGLES){
		glm::vec3 triangle[3] = { glm::vec3(curPrim.v[0].pos), glm::vec3(curPrim.v[1].pos), glm::vec3(curPrim.v[2].pos) };
		AABB aabb = getAABBForTriangle(triangle);
		//brute force baricentric 
		if (aabb.min.x<0 || aabb.max.x>width - 1 || aabb.min.y<0 || aabb.max.y>height - 1) return;
		int xmin = getMax(0, aabb.min.x);
		int xmax = getMin(aabb.max.x, width - 1);
		int ymin = getMax(0, aabb.min.y);
		int ymax = getMin(aabb.max.y, height - 1);

#if (USETEXTURE==1)
		int fixedDepth;
		for (int x = xmin; x <= xmax; x++){
			for (int y = ymin; y <= ymax; y++){ 
				int pid = x + y*width;
				//printf("pid %d\n", pid);
				glm::vec3 barcen = calculateBarycentricCoordinate(triangle, glm::vec2(x, y)); 
				if (isBarycentricCoordInBounds(barcen)){
					float zval = getZAtCoordinate(barcen, triangle);

					fixedDepth = -(int)INT_MAX*zval;
					atomicMin(&depths[pid], fixedDepth); 
					if (depths[pid] == fixedDepth ){

						Fragment & curFrag = fragments[pid];
						curFrag.eyeNor = barcen.x*curPrim.v[0].eyeNor + barcen.y*curPrim.v[1].eyeNor + barcen.z*curPrim.v[2].eyeNor;
						curFrag.eyePos = barcen.x*curPrim.v[0].eyePos + barcen.y*curPrim.v[1].eyePos + barcen.z*curPrim.v[2].eyePos;
						//add texture here
						if (vertex0.dev_diffuseTex == NULL){
							curFrag.dev_diffuseTex = NULL;
						}
						else{
							curFrag.dev_diffuseTex = vertex0.dev_diffuseTex;
						}
						curFrag.texHeight = vertex0.texHeight;
						curFrag.texWidth = vertex0.texWidth;
						curFrag.texture = vertex0.texture;
						//add color here (in case no texture)
						curFrag.color = barcen.x*curPrim.v[0].col + barcen.y*curPrim.v[1].col + barcen.z*curPrim.v[2].col;
#if USEPERSPECTIVECORRECTION==1 //https://en.wikipedia.org/wiki/Texture_mapping#Perspective_correctness for reference
						glm::vec3 tmp = glm::vec3(barcen.x / curPrim.v[0].eyePos.z, barcen.y / curPrim.v[1].eyePos.z, barcen.z / curPrim.v[2].eyePos.z);
						curFrag.texcoord0 = tmp.x*curPrim.v[0].texcoord0 + tmp.y*curPrim.v[1].texcoord0 + tmp.z*curPrim.v[2].texcoord0;
						curFrag.texcoord0 /= (tmp.x + tmp.y + tmp.z);
#else
						curFrag.texcoord0 = barcen.x*curPrim.v[0].texcoord0 + barcen.y*curPrim.v[1].texcoord0 + barcen.z*curPrim.v[2].texcoord0;

#endif
					}
				}
			}
		}
#elif  USELINES==1
		VertexOut & vertex1 = curPrim.v[1];
		VertexOut & vertex2 = curPrim.v[2];
		glm::vec3 color = vertex0.col;
		color += curPrim.v[0].eyeNor ;
		color = glm::normalize(color);
		devRasterizeLine(glm::vec3(vertex0.pos), glm::vec3(vertex1.pos), color, width, height, fragments);
		devRasterizeLine(glm::vec3(vertex0.pos), glm::vec3(vertex1.pos), color, width, height, fragments);
		devRasterizeLine(glm::vec3(vertex2.pos), glm::vec3(vertex0.pos), color, width, height, fragments);
#else
		//VertexOut & vertex1 = curPrim.v[1]; 
		thrust::minstd_rand rng; 
		thrust::uniform_real_distribution<float> dist(0, 10);
		rng.discard(randomnum);

		////int xx = vertex0.pos.x + (int)dist(rng);
		////printf("%f \n",  dist(rng));
		//int xx = vertex0.pos.x;
		//int yy = vertex0.pos.y ;
		//if (xx > 0 && xx<width &&yy>0 && yy < height){
		//	int ppid = xx + yy*width;
		//	glm::vec3 color = vertex0.col;
		//	color += curPrim.v[0].eyeNor;
		//	color = glm::normalize(color);
		//	fragments[ppid].color = color;
		//}

		//xx = vertex1.pos.x;
		//yy = vertex1.pos.y;
		//if (xx > 0 && xx<width &&yy>0 && yy < height){
		//	int ppid = xx + yy*width;
		//	glm::vec3 color = vertex1.col;
		//	color += curPrim.v[0].eyeNor;
		//	color = glm::normalize(color);
		//	fragments[ppid].color = color;
		//}
		VertexOut & vertex1 = curPrim.v[1];
		VertexOut & vertex2 = curPrim.v[2];
		glm::vec3 color = vertex0.col;
		color += curPrim.v[0].eyeNor;
		color = glm::normalize(color);
		//int stepsize = (int)glm::cos((float)randomnum);
		
		devRasterizePoints(glm::vec3(vertex0.pos), glm::vec3(vertex1.pos), color, width, height, fragments, SPARSITY);
		devRasterizePoints(glm::vec3(vertex0.pos), glm::vec3(vertex1.pos), color, width, height, fragments, SPARSITY);
		devRasterizePoints(glm::vec3(vertex2.pos), glm::vec3(vertex0.pos), color, width, height, fragments, SPARSITY);
#endif
		//}
	}
}

//__global__ void kernTileRasterize(int n, Primitive * primitives, int* depths, int width, int height, int numTiles, Tile * tiles, Fragment* fragments){
//	__shared__ unsigned int block_tile_indices[tileSizeR2];
//	__shared__  Fragment block_tile_Frags[tileSizeR2];
//
//
//
//	Tile & curTile = dev_tiles[blockIdx.x];
//	int numPrimsThisTile = curTile.numPrims;
//	int numTilesThisThread = (numPrimsThisTile + blockDim.x - 1) / blockDim.x;
//	int id0 = threadIdx.x*numTilesThisThread;
//	int upperBnd = numTilesThisThread + id0;
//	
//
//	__syncthreads();
//
//	for (int i = id0; i < upperBnd; i++){
//		if (i < numPrimsThisTile){
//			int pid; //TODO: pid = ?
//
//			Primitive & curPrim = primitives[pid];
//			VertexOut & vertex0 = curPrim.v[0];
//			glm::vec3 triangle[3] = { glm::vec3(curPrim.v[0].pos), glm::vec3(curPrim.v[1].pos), glm::vec3(curPrim.v[2].pos) };
//		}
//	}
//
//
//}
static int iter = 0;
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
	iter++;
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
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
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);  // what if rasterizer change depth, need to do depth test after rasterizer

	// TODO: rasterize 
	dim3 numBlocksPrims((totalNumPrimitives + blockSize - 1) / blockSize);
	int randomnum = std::rand();
	kernRasterize << <numBlocksPrims, blockSize >> >(totalNumPrimitives, dev_primitives, dev_depth, width, height, dev_fragmentBuffer, iter);
	checkCUDAError("rasterize wrong");

#if USETEXTURE==1
	kernTextureMap << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer);
	checkCUDAError("textur error");
#endif

	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
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
