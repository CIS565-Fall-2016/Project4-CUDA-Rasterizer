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
#include <glm/gtx/component_wise.hpp>

#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define MAX_DEPTH 10000.0f
#define DEPTH_QUANTUM (float)(INT_MAX / MAX_DEPTH)
#define getIndex(x, y, width) ((x) + (y) * (width))
#define samplesPerPixel 16
#define ambientLight 0.1
#define tuneShade 0


#define DEBUG 1
#define debug(...) if (DEBUG == 1) { printf (__VA_ARGS__); }
#define debug0(...) if (DEBUG == 1 && id == 0) { printf (__VA_ARGS__); }
#define debug1(...) if (DEBUG == 1 && id == 1) { printf (__VA_ARGS__); }
#define debugDuck(...) if (DEBUG == 1 && id == 330033) { printf (__VA_ARGS__); }
#define debugMinMax(...) if (DEBUG == 1 && id == 320) { printf (__VA_ARGS__); }
#define debugDepthsId 369228
#define debugDepths(...) if (DEBUG == 1 && id == debugDepthsId) { printf (__VA_ARGS__); }
//#define debugBoard(...) if (DEBUG == 1 && id == ) { printf (__VA_ARGS__); }

#define range(i, start, stop) for (i = start; i < stop; i++)
#define SHOW_TEXTURE 0
#define debug(...) if (DEBUG == 1) { printf (__VA_ARGS__); }

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
    glm::vec4 screenPos;

    // TODO: add new attributes to your VertexOut
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

    glm::vec3 viewPos;  // eye space position used for shading
    glm::vec3 viewNorm;  // eye space normal used for shading, cuz normal will go wrong after perspective transformation
    // glm::vec3 col;
    VertexAttributeTexcoord texcoord0;
  };

  struct Primitive {
    PrimitiveType primitiveType = Triangle; // C++ 11 init
    Vertex v[3];
    TextureData* diffuseTex = NULL;
    glm::vec2 texRes;
    // ...
  };

  struct Fragment {
    glm::vec3 color;

    // TODO: add new attributes to your Fragment
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

     glm::vec3 viewPos;  // eye space position used for shading
     glm::vec3 viewNorm;
     TextureData* diffuseTex;
	 int numSamples;
    // ...
  };

  struct VertexParts {
    int primitiveMode;  //from tinygltfloader macro
    PrimitiveType primitiveType;
    int numPrimitives;
    int numIndices;
    int numVertices;

    // Vertex, const after loaded
    VertexIndex* indices;
    VertexAttributePosition* pos;
    VertexAttributeNormal* normal;
    VertexAttributeTexcoord* texcoord0;

    // Materials, add more attributes when needed
    TextureData* diffuseTex;
    // TextureData* dev_specularTex;
    // TextureData* dev_normalTex;
    // ...

    // Vertex Out, vertex used for rasterization, this is changing every frame
    Vertex* vertices;

    // TODO: add more attributes when needed
    glm::vec2 texRes;
  };

}

static std::map<std::string, std::vector<VertexParts>> mesh2vertexParts;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static unsigned int *dev_depth = NULL;  // you might need this buffer when doing depth test

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
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
  width = w;
  height = h;
  int numSamples = samplesPerPixel * width * height;

  cudaFree(dev_fragmentBuffer);
  cudaMalloc(&dev_fragmentBuffer, numSamples * sizeof(Fragment));
  cudaMemset(dev_fragmentBuffer, 0, numSamples * sizeof(Fragment));

  cudaFree(dev_framebuffer);
  cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
  cudaFree(dev_depth);
  cudaMalloc(&dev_depth, numSamples * sizeof(unsigned int));

  checkCUDAError("rasterizeInit");
}

__global__
void _initDepth(int length, int *depth)
{
  if (IDx < length * samplesPerPixel)
  {
    depth[IDx] = INT_MAX;
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

          // malloc for Vertex
          Vertex* dev_vertex;
          cudaMalloc(&dev_vertex, numVertices * sizeof(Vertex));
          checkCUDAError("Malloc VertexOut Buffer");

          // ----------Materials-------------

          // You can only worry about this part once you started to 
          // implement textures for your rasterizer
          TextureData* dev_diffuseTex = NULL;
          glm::vec2 texRes;
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
                  texRes = glm::vec2(image.width, image.height);

                  checkCUDAError("Set Texture Image data");
                }
              }
            }

            // TODO: write your code for other materials
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

            dev_vertex,  //VertexOut
            texRes
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

//////////////// PIPELINE CODE ////////////////


__global__ 
void _vertexTransformAndAssembly(
  const int numVertices, 
  VertexParts vertexParts, 
  const glm::mat4 MVP, const glm::mat4 MV, const glm::mat3 MV_normal, 
  const int width, const int height) {

  // vertex id
  if (IDx >= numVertices) return;

  Vertex &vertex = vertexParts.vertices[IDx];

  // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
  // Then divide the pos by its w element to transform into NDC space
  // Finally transform x and y to viewport space
  // TODO: Apply vertex transformation here
  glm::vec4 modelPos = glm::vec4(vertexParts.pos[IDx], 1); // this is in model space
  vertex.viewPos = glm::vec3(MV * modelPos);
  vertex.viewNorm = glm::vec3(MV_normal * vertexParts.normal[IDx]);
  glm::vec4 clipPos(MVP * modelPos);
  glm::vec4 screenDims(width, height, 1, 1);
  vertex.screenPos = screenDims * (clipPos / clipPos.w + glm::vec4(1, 1, 1, 0)) / 2.0f;

  // Assemble all attribute arrays into the primitive array
  vertex.texcoord0 = vertexParts.texcoord0[IDx];
}



static int curPrimitiveBeginId = 0;

// START HERE: figure out where to put texture info

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* primitives, VertexParts vertexParts) {

  // index id
  if (IDx < numIndices) {

    // TODO: uncomment the following code for a start
//This is primitive assembly for triangles

if (vertexParts.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
  int n_vertices = vertexParts.primitiveType;
  int prim_id = IDx / n_vertices + curPrimitiveBeginId;
  Primitive &primitive = primitives[prim_id];
  primitive.v[IDx % n_vertices]
    = vertexParts.vertices[vertexParts.indices[IDx]];
  primitive.diffuseTex = vertexParts.diffuseTex;
  primitive.texRes = vertexParts.texRes;
}
// TODO: other primitive types (point, line)
  }

}

__device__
float getFragmentDepth(glm::vec3 bcCoord, glm::vec3 tri[3]) {
  // get depth of fragment represented as an integer
  int i;
  float depth = 0;

  // interpolate vertex depths
  range(i, 0, 3) {
    depth += bcCoord[i] * tri[i].z;
  }
  if (depth > 1) {
    return INT_MAX;
  }
  else if (depth < 0) {
    return 0;
  }
  else {
    return depth * INT_MAX;
  };
}

__device__
 void atomicAddVec3(glm::vec3 &vec1, glm::vec3 vec2) {
  int i;
  range(i, 0, 3) {
    atomicAdd(&vec1[i], vec2[i]);
  }
}

__global__
void _rasterize(int n_primitives, int height, int width,
const Primitive *primitives, unsigned int *depths, Fragment *fragments) {
  if (IDx >= n_primitives) return;
  int id = IDx;

  int i, y, x, offset;
  Primitive primitive = primitives[IDx];
  glm::vec3 tri[3];
  range(i, 0, 3) {
    // get coordinates of tri points
    tri[i] = glm::vec3(primitive.v[i].screenPos);
  }

  thrust::uniform_real_distribution<float> u01(0, 1);
  AABB aabb = getAABBForTriangle(tri);
  range(y, aabb.min.y, aabb.max.y) {
    range(x, aabb.min.x, aabb.max.x) {
      thrust::default_random_engine seed(getIndex(x, y, width));
      range(offset, 0, samplesPerPixel) {
        int index = samplesPerPixel * getIndex(width - x, height - y, width) + offset;
        glm::vec2 screenPos = glm::vec2(x + u01(seed), y + u01(seed));

        // determine if screenPos is inside polygon
        glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, screenPos);
        if (isBarycentricCoordInBounds(barycentricCoord)) {
          unsigned int depth = getFragmentDepth(barycentricCoord, tri);

          // assign fragEyePos.z to dev_depth[i] iff it is smaller 
          // (fragment is closer to camera) 
          atomicMin(depths + index, depth);
        }
      }
    }
  }

  __syncthreads(); // wait for all depths to be updated


  range(y, aabb.min.y, aabb.max.y) {
    range(x, aabb.min.x, aabb.max.x) {
      thrust::default_random_engine seed(getIndex(x, y, width));
      range(offset, 0, samplesPerPixel) {
        int index = samplesPerPixel *  getIndex(width - x, height - y, width) + offset;
        glm::vec2 screenPos = glm::vec2(x + u01(seed), y + u01(seed));

        // determine if screenPos is inside polygon
        glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, screenPos);
        if (isBarycentricCoordInBounds(barycentricCoord)) {
          float depth = getFragmentDepth(barycentricCoord, tri);

          // if the sample is not occluded
          if ((unsigned int)depth == depths[index]) {
            //debug(".");
            //debugDepths("depth=%u\n", depth);
            Fragment &fragment = fragments[index];


            // interpolate texcoord and viewPos and texnorm
            fragment.viewPos = glm::vec3(0);
            fragment.viewNorm = glm::vec3(0);
            glm::vec2 texcoord(0);
            float texWeightNorm = 0;
            range(i, 0, 3) {
              float weight = barycentricCoord[i];
              Vertex v = primitive.v[i];

              fragment.viewNorm += weight * v.viewNorm;
              fragment.viewPos += weight * v.viewPos;

              float texWeight = weight / v.viewPos.z;
              texcoord += texWeight * v.texcoord0;
              texWeightNorm += texWeight;
            }

            // get the color using texcoord
            texcoord /= texWeightNorm;
            glm::vec2 texRes = primitive.texRes;
            glm::vec2 scaledCoord = texcoord * glm::vec2(texRes.x, texRes.y);
            int tid = 3 * getIndex((int)scaledCoord.x, (int)scaledCoord.y, texRes.x);
            TextureData *tex = primitive.diffuseTex; 
            glm::vec3 color = glm::vec3(tex[tid + 0], tex[tid + 1], tex[tid + 2]) / 255.0f;

            //atomicAddVec3(fragment.color, color);
            fragment.color = color;
          }
        }
      }
    }
  }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void _render(int w, int h, const Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
  if (IDx >= w || IDy >= h) return;
  int index = getIndex(IDx, IDy, w);
  int offset;
  range(offset, 0, samplesPerPixel) {
    int sampleId = samplesPerPixel * index + offset;
    Fragment frag = fragmentBuffer[sampleId];
    glm::vec3 lightPos(0);
    glm::vec3 L = glm::normalize(glm::vec3(0, -1, -1));//lightPos - frag.viewPos);
    glm::vec3 V = glm::normalize(-frag.viewPos);
    glm::vec3 H = glm::normalize(L + V);
    float intensity = saturate(glm::dot(frag.viewNorm, H) + 0.2) + ambientLight;
    if (tuneShade) {
      intensity = intensity > 0.5 ? 1 : ambientLight;
    }
    framebuffer[index] += intensity * frag.color / (float)samplesPerPixel;
  }
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength = 8;
    dim3 blockSize2d(sideLength, sideLength);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
    (height - 1) / blockSize2d.y + 1);

  // Execute your rasterization pipeline here
  cudaMemset(dev_primitives, 0, width * height * sizeof(Primitive));
  cudaMemset(dev_fragmentBuffer, 0, samplesPerPixel * width * height * sizeof(Fragment));

  // (See README for rasterization pipeline outline.)

  // Vertex Process & primitive assembly
      
  dim3 numThreadsPerBlock(128);
  curPrimitiveBeginId = 0;
  {

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
        checkCUDAError("Vertex Transform and Assembly");
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
  
  //_initDepth << <blockSize, sampleBlockSize2d >> >(numSamples, dev_depth);
  cudaMemset(dev_depth, 0xff, samplesPerPixel * width * height * sizeof(dev_depth[0]));
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(dev_framebuffer[0]));
  
  // TODO: rasterize
  dim3 blockSize = totalNumPrimitives / numThreadsPerBlock.x + 1;
  _rasterize<< <blockSize, numThreadsPerBlock>> > 
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
      cudaFree(p->indices);
      cudaFree(p->pos);
      cudaFree(p->normal);
      cudaFree(p->texcoord0);
      cudaFree(p->diffuseTex);
      cudaFree(p->vertices);

      
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
