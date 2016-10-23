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

#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define MAX_DEPTH 10000.0f
#define DEPTH_QUANTUM (float)(INT_MAX / MAX_DEPTH)

#define DEBUG 1
#define debug(...) if (DEBUG == 1) { printf (__VA_ARGS__); }
#define debug0(...) if (DEBUG == 1 && id == 0) { printf (__VA_ARGS__); }
#define range(i, start, stop) for (i = start; i < stop; i++)

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

  struct Vertex {
    glm::vec4 pos;

    // TODO: add new attributes to your VertexOut
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

     glm::vec3 eyePos;  // eye space position used for shading
     glm::vec3 eyeNor;  // eye space normal used for shading, cuz normal will go wrong after perspective transformation
    // glm::vec3 col;
  };

  struct Primitive {
    PrimitiveType primitiveType = Triangle; // C++ 11 init
    Vertex v[3];
    glm::vec2 texcoord0;
    TextureData* dev_diffuseTex = NULL;
    // int texWidth, texHeight;
    // ...
  };

  struct Fragment {
    glm::vec3 color;

    // TODO: add new attributes to your Fragment
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

     glm::vec3 eyePos;  // eye space position used for shading
     glm::vec3 eyeNor;
     VertexAttributeTexcoord texcoord0;
     TextureData* dev_diffuseTex;
    // ...
  };

  struct VertexParts {
    int primitiveMode;  //from tinygltfloader macro
    PrimitiveType primitiveType;
    int numPrimitives;
    int numIndices;
    int numVertices;

    // Vertex, const after loaded
    VertexIndex* dev_indices;
    VertexAttributePosition* dev_position;
    VertexAttributeNormal* dev_normal;
    VertexAttributeTexcoord* dev_texcoord0;

    // Materials, add more attributes when needed
    TextureData* dev_diffuseTex;
    // TextureData* dev_specularTex;
    // TextureData* dev_normalTex;
    // ...

    // Vertex Out, vertex used for rasterization, this is changing every frame
    Vertex* dev_vertices;

    // TODO: add more attributes when needed
  };

}

static std::map<std::string, std::vector<VertexParts>> mesh2vertexParts;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int *dev_depth = NULL;  // you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void _sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
	int x = IDx;
	int y = IDy;
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
void _render(int w, int h, const Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
	int x = IDx;
	int y = IDy;
    int index = x + (y * w);

    if (x >= w || y >= h) return;
	framebuffer[index] = fragmentBuffer[index].color;
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

__global__
void _initDepth(int w, int h, int *depth)
{
  if (IDx < w && IDy < h)
  {
    int index = IDx + (IDy * w);
    depth[index] = FLT_MAX;
  }
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(
int N, BufferByte* dev_dst, const BufferByte* dev_src, 
int n, int byteStride, int byteOffset, int componentTypeByteSize) {
  
  // Attribute (vec3 position)
  // component (3 * float)
  // byte (4 * byte)

  // id of component
  if (IDx < N) {
    int count = IDx / n;
    int offset = IDx - count * n; // which component of the attribute

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
  if (IDx < numVertices) {
    position[IDx] = glm::vec3(MV * glm::vec4(position[IDx], 1.0f));
    normal[IDx] = glm::normalize(MV_normal * normal[IDx]);
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
  //    for each primitive: 
  //      build device buffer of indices, materail, and each attributes
  //      and store these pointers in a map
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

        auto res = mesh2vertexParts.insert(std::pair<std::string, std::vector<VertexParts>>(mesh.name, std::vector<VertexParts>()));
        std::vector<VertexParts> & vertexPartsVector = (res.first)->second;

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

          // malloc for Vertex
          Vertex* dev_vertex;
          cudaMalloc(&dev_vertex, numVertices * sizeof(Vertex));
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
                  // image.width;
                  // image.height;

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
          vertexPartsVector.push_back(VertexParts{
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

            dev_vertex  //VertexOut
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
  const int numVertices, 
  VertexParts vertexParts, 
  const glm::mat4 MVP, const glm::mat4 MV, const glm::mat3 MV_normal, 
  const int width, const int height) {

  // vertex id
  int vid = IDx;
  int id = vid;
  if (vid >= numVertices) return;

  Vertex &vertex = vertexParts.dev_vertices[vid];

  // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
  // Then divide the pos by its w element to transform into NDC space
  // Finally transform x and y to viewport space
  glm::vec4 modelPos = glm::vec4(vertexParts.dev_position[vid], 1.0); // this is in model space
  vertex.pos = (float)width / 2.0f * (modelPos / modelPos.w + 1.0f);
  // TODO go over this carefully

  // TODO: Apply vertex transformation here

  // Assemble all attribute arrays into the primitive array
  //vertex.eyePos = vertexParts.dev_normal[vid];
  //vertex.eyeNor = vertexParts.dev_normal[vid];
  //vertex.texcoord0 = vertexParts.dev_texcoord0[vid];
  //vertex.dev_diffuseTex = vertexParts.dev_diffuseTex;
}



static int curPrimitiveBeginId = 0;

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, VertexParts vertexParts) {

  // index id
  int iid = IDx;

  if (iid < numIndices) {

    // TODO: uncomment the following code for a start
     //This is primitive assembly for triangles

    int pid;	// id for cur primitives vector
    if (vertexParts.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
    	pid = iid / (int)vertexParts.primitiveType;
    	dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)vertexParts.primitiveType]
    		= vertexParts.dev_vertices[vertexParts.dev_indices[iid]];
    }
    int id = IDx; //START HERE

    // TODO: other primitive types (point, line)
  }

}

__device__
glm::vec3 intersect(glm::vec3 vector, glm::mat3 plane) {
  glm::vec3 x = glm::inverse(plane) * vector;
  float t = 1 / glm::dot(x, glm::vec3(1.0f));
  return t * vector;
}


__device__
int getFragmentDepth(int x, int y, glm::vec3 tri[3]) {

  // test if fragment is in primitive
  glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
  if (isBarycentricCoordInBounds(barycentricCoord)) {
    // fragment is in primitive

    // get depth of fragment represented as an integer
    int i;
    float depth = 0;
    range(i, 0, 3) {
      glm::vec3 weighted = barycentricCoord[i] * tri[i];
      depth += weighted.z;
    }
    return depth > MAX_DEPTH ? INT_MAX : (int)(depth * DEPTH_QUANTUM);
  }
  else {
    return INT_MAX;
  }

}

__device__ 
int getIndex(int i, int j, int width) {
  return i * width + j;
}

__global__
void test() {
  debug("test ");
}


__global__
void _rasterize(int n_primitives, int height, int width,
const Primitive *primitives, int *depths, Fragment *fragments) {
  if (IDx >= n_primitives) return;

  int i, j;
  Primitive primitive = primitives[IDx];
  glm::vec3 tri[3];
  range(i, 0, 3) { 
    // get coordinates of tri points
    tri[i] = glm::vec3(primitive.v[i].pos);
  }

  range(i, 0, height) { 
    range(j, 0, width) {
      int index = getIndex(i, j, width);

		  glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
      if (isBarycentricCoordInBounds(barycentricCoord)) {
        int depth = getFragmentDepth(i, j, tri);

        // assign fragEyePos.z to dev_depth[i] iff it is smaller 
        // (fragment is closer to camera)
        int index = getIndex(i, j, width);
        atomicMin(depths + index, depth);
      }
    }
  }

  __syncthreads(); // wait for all depths to be updated

  range(i, 0, height) {
    range(j, 0, width) { 
      int index = getIndex(i, j, width);
      glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
      if (isBarycentricCoordInBounds(barycentricCoord)) {

        int depth = getFragmentDepth(i, j, tri);
        if (depth == depths[index]) {
          Fragment &fragment = fragments[index];
          //Vertex vertex = primitive.v[0]; // TODO: move common fields into Primitive

          //fragment.dev_diffuseTex = vertex.dev_diffuseTex;
          //fragment.eyeNor = vertex.eyeNor;
          //fragment.eyePos = vertex.eyePos;
          //fragment.texcoord0 = vertex.texcoord0;

          ////TODO: get rid of this
          fragment.color = glm::vec3(1.0f);
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
      
  curPrimitiveBeginId = 0;
  {
    dim3 numThreadsPerBlock(128);

    auto it = mesh2vertexParts.begin();
    auto itEnd = mesh2vertexParts.end();

    for (; it != itEnd; ++it) {
      auto parts = (it->second).begin();  // each primitive
      auto partsEnd = (it->second).end();
      for (; parts != partsEnd; ++parts) {
        dim3 numBlocksForVertices((parts->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
        dim3 numBlocksForIndices((parts->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

        _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >
          (parts->numVertices, 
          *parts, 
          MVP, MV, MV_normal, 
          width, height);
        checkCUDAError("Vertex Processing");
        cudaDeviceSynchronize();
        _primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
          (parts->numIndices, 
          curPrimitiveBeginId, 
          dev_primitives, 
          *parts);
        checkCUDAError("Primitive Assembly");

        curPrimitiveBeginId += parts->numPrimitives;
      }
    }

    checkCUDAError("Vertex Processing and Primitive Assembly");
  }
  
  int numPixels = width * height;
  _initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
  
  // TODO: rasterize
  _rasterize<< <blockCount2d, blockSize2d >> > 
    (totalNumPrimitives, height, width, 
    dev_primitives, dev_depth, dev_fragmentBuffer);
  checkCUDAError("rasterizer");

  // Copy depthbuffer colors into framebuffer
  _render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
  checkCUDAError("fragment shader");
  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  _sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
  checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

  auto it(mesh2vertexParts.begin());
  auto itEnd(mesh2vertexParts.end());
  for (; it != itEnd; ++it) {
    for (auto p = it->second.begin(); p != it->second.end(); ++p) {
      cudaFree(p->dev_indices);
      cudaFree(p->dev_position);
      cudaFree(p->dev_normal);
      cudaFree(p->dev_texcoord0);
      cudaFree(p->dev_diffuseTex);
      cudaFree(p->dev_vertices);

      
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
