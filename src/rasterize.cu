/**__global__ void clearVertexBuffer(int n, Fragment* dev_fragmentBuffer, glm::vec3 color)
{
int index = (blockIdx.x * blockDim.x) + threadIdx.x;

if (index < n)
{
dev_fragmentBuffer[index].color = color;
}
}
* @file      rasterize.cu
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
* @date      2012-2016is
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
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

static glm::vec4 lightpos(500.0f, 500.0f, 500.0f, 1.0f);
static glm::vec4 lightcol(0.92f, 0.92f, 0.85f, 1.0f);
static int numlights = 4;
//namespace {

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
        //glm::vec3 col;
        glm::vec2 texcoord0;
        TextureData* dev_diffuseTex = NULL;
        int texWidth, texHeight;
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
        // VertexAttributeTexcoord texcoord0;
        TextureData* dev_diffuseTex;
        float depth;
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
        // TextureData* dev_specularTex;
        // TextureData* dev_normalTex;
        // ...

        // Vertex Out, vertex used for rasterization, this is changing every frame
        VertexOut* dev_verticesOut;

        // TODO: add more attributes when needed
        int txWidth;
        int txHeight;
    };

//}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static int cwidth = 0;
static int cheight = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Primitive *dev_primitivestmp = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static Fragment *dev_dsfragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *dev_dsframebuffer = NULL;

static glm::vec4 *dev_lightspos = NULL;
static glm::vec4 *dev_lightscol = NULL;


static int * dev_depth = NULL;	// you might need this buffer when doing depth test


__host__ __device__ bool operator<(const Primitive &lhs, const Primitive &rhs)
{
    return (lhs.v[0].eyePos.z + lhs.v[1].eyePos.z + lhs.v[2].eyePos.z) > (rhs.v[0].eyePos.z + rhs.v[1].eyePos.z + rhs.v[2].eyePos.z);
}


struct is_backface
{
    __host__ __device__
        bool operator()(const Primitive p)
    {
        return (p.v[0].eyeNor.z < 0.0 && p.v[2].eyeNor.z < 0.0 && p.v[2].eyeNor.z < 0.0);
    }
};

__global__
void downsample(int w, int h, glm::vec3 *dev_framebuffer, glm::vec3 *dev_dsframebuffer, bool aa) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int index = x + (y * w);

    if (x < w - 1 && y < h - 1)
    {
        int index2 = ((x)* 2) + ((y)* 4 * (w));
        int index3 = x * 2 + ((y + 1) * 4 * w);

        if (aa)
        {
            dev_dsframebuffer[index] =
                (dev_framebuffer[index2] +
                dev_framebuffer[index2 + 1] +
                dev_framebuffer[index3] +
                dev_framebuffer[index3 + 1]
                ) * 0.25f;
        }
        else
            dev_dsframebuffer[index] = dev_framebuffer[index2];
    }
}




//
// Kernel that writes the image to the OpenGL PBO directly.
//
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

//
// Writes fragment colors to the framebuffer
//
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, glm::vec4 lightposition, glm::vec4 lightcolor, bool spec) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = fragmentBuffer[index].color;
        
        
        // TODO: add your fragment shader code here
        glm::vec3 incidentvec = glm::normalize(fragmentBuffer[index].eyePos - glm::vec3(lightposition));
        
        // gouraud
        framebuffer[index] *= glm::dot(incidentvec, -fragmentBuffer[index].eyeNor) * glm::vec3(lightcolor);
        //if (index == 2500)
        //    printf("incidentvec = %f %f %f\n", incidentvec.x, incidentvec.y, incidentvec.z);
        
        // spec
        if (spec)
        {
            float speccontrib = 0.5;
            int power = 2;
            glm::vec3 incident = glm::normalize(fragmentBuffer[index].eyePos);
            glm::vec3 ilight = glm::normalize(glm::vec3(lightposition) - fragmentBuffer[index].eyePos);
            glm::vec3 rfl = glm::reflect(incident, fragmentBuffer[index].eyeNor);
            framebuffer[index] += powf(glm::clamp(glm::dot(rfl, ilight), 0.0f, 1.0f), power) * speccontrib * glm::vec3(lightcolor);
        }
        
    }
}

//
// Called once at the beginning of the program to allocate memory.
//
void rasterizeInit(int w, int h, int cw, int ch) {
    width = w;
    height = h;
    cwidth = cw;
    cheight = ch;
    cudaFree(dev_fragmentBuffer);
    cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaMalloc(&dev_dsfragmentBuffer, width * height * sizeof(Fragment));
    cudaMemset(dev_dsfragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    cudaFree(dev_dsframebuffer);
    cudaMalloc(&dev_dsframebuffer, cwidth * cheight * sizeof(glm::vec3));
    cudaMemset(dev_dsframebuffer, 0, cwidth * cheight * sizeof(glm::vec3));

    cudaFree(dev_lightspos);
    cudaMalloc(&dev_lightspos, numlights * sizeof(glm::vec4));
    cudaMemset(dev_lightspos, 0, numlights * sizeof(glm::vec4));

    cudaFree(dev_lightscol);
    cudaMalloc(&dev_lightscol, numlights * sizeof(glm::vec4));
    cudaMemset(dev_lightscol, 0, numlights * sizeof(glm::vec4));

    cudaFree(dev_depth);
    cudaMalloc(&dev_depth, width * height * sizeof(int));

    checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, Fragment* dev_fragmentbuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        dev_fragmentbuffer[index].depth = 999999999.9f;
    }
}


__global__
void mergeframebuffers(int w, int h, Fragment* dev_out, Fragment* dev_in, float div)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        dev_out[index].color = (dev_out[index].color + dev_in[index].color) * div;
    }
}

__global__
void multframebuffer(int w, int h, Fragment* dev_in, float m)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        dev_in[index].color = dev_in[index].color * m;
    }
}

__global__
void accumframebuffers(int w, int h, Fragment* dev_dst, Fragment* dev_src)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        dev_dst[index].color += dev_src[index].color;
        //dev_dst[index].dev_diffuseTex = dev_src[index].dev_diffuseTex;
        dev_dst[index].eyeNor = (dev_dst[index].eyeNor + dev_src[index].eyeNor)*0.5f;
        dev_dst[index].eyePos = (dev_dst[index].eyePos + dev_src[index].eyePos)*0.5f;
        dev_dst[index].depth = (dev_dst[index].depth + dev_src[index].depth)*0.5f;
    }
}


//
// kern function with support for stride to sometimes replace cudaMemcpy
// One thread is responsible for copying one component
//
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

                    dim3 numThreadsPerBlock(32);
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

                        dim3 numThreadsPerBlock(32);
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

                    int width = 0;
                    int height = 0;

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
                                    width = image.width;
                                    height = image.height;

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

                        dev_vertexOut,	//VertexOut

                        width,
                        height,
                    });

                    totalNumPrimitives += numPrimitives;

                } // for each primitive

            } // for each mesh

        } // for each node

    }


    // 3. Malloc for dev_primitives
    {
        cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
        cudaMalloc(&dev_primitivestmp, totalNumPrimitives * sizeof(Primitive));
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

        // TODO: Apply vertex transformation here
        // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
        // Then divide the pos by its w element to transform into NDC space
        // Finally transform x and y to viewport space


        glm::vec4 p = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
        glm::vec4 ep = MV * glm::vec4(primitive.dev_position[vid], 1.0f);

        //if (vid % 8 == 0)
        //    printf("\n[%f %f] [%f %f] [%f %f]", p.x, p.y, p.x, p.y, p.x, p.y);

        p /= p.w;

        p.x = 0.5 * width * ((-p.x) * p.w + 1.0f);
        p.y = 0.5 * height * ((-p.y) * p.w + 1.0f);

        // TODO: Apply vertex assembly here
        // Assemble all attribute arraies into the primitive array

        primitive.dev_verticesOut[vid].pos = p;


        glm::vec3 n = MV_normal * primitive.dev_normal[vid];
        primitive.dev_verticesOut[vid].eyeNor = n;
        primitive.dev_verticesOut[vid].eyePos = glm::vec3(ep);
        //primitive.dev_verticesOut[vid].col = primitive.col;

        if (primitive.dev_texcoord0 != NULL)
        {
            primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
            primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
        }
    }
}

__global__ void clearVertexBuffer(int n, Fragment* dev_fragmentBuffer, glm::vec3 color)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < n)
    {
        dev_fragmentBuffer[index].color = color;
        dev_fragmentBuffer[index].depth = 999999999.9;
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
        // processed in scanline function
    }

}

__global__
void scanline(int w, int h, Fragment* dev_fragBuffer, int numidx, int idbegin,
Primitive* dev_primitives, PrimitiveDevBufPointers primitive,
int mode, bool perspectivecorrect, float xoffset, float yoffset, bool aabbcheck, bool cheapculling)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        //printf("\nnum idx = %d", primitive.numPrimitives);
        //primitive.dev_verticesOut[primitive.dev_indices[index]]

        //printf("\n1");
        glm::vec2 coords(x + xoffset, y + yoffset);
        
        float depth = dev_fragBuffer[index].depth;
        for (int i = 0; i < numidx; i ++)
        {
            //hit = false;
            
            glm::vec3 p[] = { glm::vec3(dev_primitives[i].v[0].pos),
                glm::vec3(dev_primitives[i].v[1].pos),
                glm::vec3(dev_primitives[i].v[2].pos) };


            if (aabbcheck)
            {
                // bounding box check
                float eps = 0.2f;
                float minx = fminf(p[0].x - eps, fminf(p[1].x - eps, p[2].x - eps));
                float maxx = fmaxf(p[0].x + eps, fmaxf(p[1].x + eps, p[2].x + eps));
                float miny = fminf(p[0].y - eps, fminf(p[1].y - eps, p[2].y - eps));
                float maxy = fmaxf(p[0].y + eps, fmaxf(p[1].y + eps, p[2].y + eps));

                if (minx > x || maxx < x || miny > y || maxy < y)
                    continue;
                // bounding box end
            }
            /*
            // remove backfaces without thrust
            if (dev_primitives[i].v[0].eyeNor.z < 0.0f && 
                dev_primitives[i].v[1].eyeNor.z < 0.0f && 
                dev_primitives[i].v[2].eyeNor.z < 0.0f)
                continue;
            */

            //printf("\npos[0] = %f %f %f", p[0].x, p[0].y, p[0].z);

            //printf("2");
            glm::vec3 bc = calculateBarycentricCoordinate(p, coords);

            //printf("\npos[0] = %f %f %f", bc.x, bc.y, bc.z);
            //if (coords.y > 100.0f)
            //    printf("coords = %f %f\n", coords.x, coords.y);
            if (bc.x <  0.0 || bc.x > 1.0 ||
                bc.y <  0.0 || bc.y > 1.0 ||
                bc.z <  0.0 || bc.z > 1.0)
            {
                continue;
            }
            else
            {
                //printf("\nHELLO WE ARE HERE!");
                if (mode == 0)  // polygons
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { dev_primitives[i].v[0].eyeNor,
                            dev_primitives[i].v[1].eyeNor,
                            dev_primitives[i].v[2].eyeNor };


                        glm::vec3 e[] = { dev_primitives[i].v[0].eyePos,
                            dev_primitives[i].v[1].eyePos,
                            dev_primitives[i].v[2].eyePos };


                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;


                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);
                        depth = tmp_depth;
                        //dev_fragBuffer[index].color = glm::vec3(1.0, 0.0, 0.0);


                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;


                        // COLOR TEST --------------------------------------------
                        if (dev_primitives[0].v[0].dev_diffuseTex == NULL)
                        {
                            dev_fragBuffer[index].color = glm::vec3(0.8, 0.8, 0.8);
                            dev_fragBuffer[index].depth = depth;
                            continue;
                        }
                        // COLOR TEST --------------------------------------------

                        float dt = glm::dot(glm::normalize(ee - pp), nn);
                        //dev_fragBuffer[index].color *= dt;


                        // texture test

                        glm::vec3 uvs[] = { glm::vec3(dev_primitives[i].v[0].texcoord0, 0.0f),
                            glm::vec3(dev_primitives[i].v[1].texcoord0, 0.0f),
                            glm::vec3(dev_primitives[i].v[2].texcoord0, 0.0f) };

                        glm::vec3 uv(0.0f);
                        if (perspectivecorrect)
                        {

                            float w = (1.0f / p[0].z) * bc[0] + (1.0f / p[1].z) * bc[1] + (1.0f / p[2].z) * bc[2];
                            float u = ((uvs[0].x / p[0].z) * bc[0] + (uvs[1].x / p[1].z) * bc[1] + (uvs[2].x / p[2].z) * bc[2]) / w;
                            float v = ((uvs[0].y / p[0].z) * bc[0] + (uvs[1].y / p[1].z) * bc[1] + (uvs[2].y / p[2].z) * bc[2]) / w;
                            uv.x = u;
                            uv.y = v;

                            /*
                            uvs[0] /= p[0].z;
                            uvs[1] /= p[1].z;
                            uvs[2] /= p[2].z;

                            uvs[0].z = 1.0f / p[0].z;
                            uvs[1].z = 1.0f / p[1].z;
                            uvs[2].z = 1.0f / p[2].z;
                            */

                        }
                        else
                        {
                            uv = (uvs[0] * bc.x + uvs[1] * bc.y + uvs[2] * bc.z);
                        }

                        /*
                        if (perspectivecorrect)
                        {
                        float zz = 1.0f / (uvs[0].z * bc.x + uvs[1].z * bc.y + uvs[2].z * bc.z);
                        uv *= zz;
                        }
                        */
                        uv *= primitive.txHeight;

                        int cix = ((int)uv.y * primitive.txHeight + (int)uv.x) * 3;
                        cix = cix % (primitive.txHeight * primitive.txWidth * 3);


                        //primitive.dev_verticesOut[primitive.dev_indices[i]].texcoord0;
                        unsigned char tx1 = dev_primitives[0].v[0].dev_diffuseTex[cix];
                        unsigned char tx2 = dev_primitives[0].v[0].dev_diffuseTex[cix + 1];
                        unsigned char tx3 = dev_primitives[0].v[0].dev_diffuseTex[cix + 2];

                        unsigned int red = tx1;
                        unsigned int green = tx2;
                        unsigned int blue = tx3;


                        //finalcolor = glm::vec3((float)red / 255.0, (float)green / 255.0, (float)blue / 255.0);
                        dev_fragBuffer[index].color = glm::vec3((float)red / 255.0, (float)green / 255.0, (float)blue / 255.0);
                        dev_fragBuffer[index].depth = depth;
                    }
                }
                else if (mode == 1 && (bc.x <= 0.04f || bc.y <= 0.04f || bc.z <= 0.04f))
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { dev_primitives[i].v[0].eyeNor,
                            dev_primitives[i].v[0].eyeNor,
                            dev_primitives[i].v[0].eyeNor };


                        glm::vec3 e[] = { dev_primitives[i].v[0].eyePos,
                            dev_primitives[i].v[0].eyePos,
                            dev_primitives[i].v[0].eyePos };

                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);

                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;

                        depth = tmp_depth;
                        dev_fragBuffer[index].color = glm::vec3(1.0, 1.0, 1.0);
                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;

                        float dt = glm::dot(glm::normalize(ee - pp), nn);
                        dev_fragBuffer[index].color *= dt;
                        dev_fragBuffer[index].depth = depth;
                    }
                }
                
                else if ((bc.x <= 0.04f && bc.y <= 0.04f) ||
                         (bc.x <= 0.04f && bc.z <= 0.04f) ||
                         (bc.y <= 0.04f && bc.z <= 0.04f))
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { dev_primitives[i].v[0].eyeNor,
                            dev_primitives[i].v[0].eyeNor,
                            dev_primitives[i].v[0].eyeNor };


                        glm::vec3 e[] = { dev_primitives[i].v[0].eyePos,
                            dev_primitives[i].v[0].eyePos,
                            dev_primitives[i].v[0].eyePos };

                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);

                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;


                        depth = tmp_depth;
                        dev_fragBuffer[index].color = glm::vec3(1.0, 1.0, 1.0);
                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;

                        float dt = glm::dot(glm::normalize(ee - pp), nn);
                        dev_fragBuffer[index].color *= dt;
                        dev_fragBuffer[index].depth = depth;

                    }
                }
                if (cheapculling) // not advised
                    break;  // dangerous bold move used after sorting that assumes that no bigger polygons intersect
            }
            //printf("3");
        }
    }
}


__global__
void scanline_bak(int w, int h, Fragment* dev_fragBuffer, int numidx, int idbegin,
Primitive* dev_primitives, PrimitiveDevBufPointers primitive,
int mode, bool perspectivecorrect, float xoffset, float yoffset)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        int index = x + (y * w);
        //printf("\nnum idx = %d", primitive.numPrimitives);
        //primitive.dev_verticesOut[primitive.dev_indices[index]]

        //printf("\n1");
        glm::vec2 coords(x + xoffset, y + yoffset);

        float depth = dev_fragBuffer[index].depth;
        for (int i = 0; i < primitive.numIndices; i += 3)
        {
            //hit = false;
            glm::vec3 p[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].pos,
                (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].pos,
                (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].pos };

            //printf("2");
            glm::vec3 bc = calculateBarycentricCoordinate(p, coords);

            if (bc.x <  0.0 || bc.x > 1.0 ||
                bc.y <  0.0 || bc.y > 1.0 ||
                bc.z <  0.0 || bc.z > 1.0)
            {
                continue;
            }
            else
            {
                if (mode == 0)
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyeNor };


                        glm::vec3 e[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyePos };

                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);

                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;


                        depth = tmp_depth;
                        //dev_fragBuffer[index].color = glm::vec3(1.0, 0.0, 0.0);
                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;
                 
                        float dt = glm::dot(glm::normalize(ee - pp), nn);
                        //dev_fragBuffer[index].color *= dt;


                        // texture test

                        glm::vec3 uvs[] = { glm::vec3(primitive.dev_texcoord0[primitive.dev_indices[i]], 0.0f),
                            glm::vec3(primitive.dev_texcoord0[primitive.dev_indices[i + 1]], 0.0f),
                            glm::vec3(primitive.dev_texcoord0[primitive.dev_indices[i + 2]], 0.0f) };

                        glm::vec3 uv(0.0f);
                        if (perspectivecorrect)
                        {

                            float w = (1.0f / p[0].z) * bc[0] + (1.0f / p[1].z) * bc[1] + (1.0f / p[2].z) * bc[2];
                            float u = ((uvs[0].x / p[0].z) * bc[0] + (uvs[1].x / p[1].z) * bc[1] + (uvs[2].x / p[2].z) * bc[2]) / w;
                            float v = ((uvs[0].y / p[0].z) * bc[0] + (uvs[1].y / p[1].z) * bc[1] + (uvs[2].y / p[2].z) * bc[2]) / w;
                            uv.x = u;
                            uv.y = v;

                            /*
                            uvs[0] /= p[0].z;
                            uvs[1] /= p[1].z;
                            uvs[2] /= p[2].z;

                            uvs[0].z = 1.0f / p[0].z;
                            uvs[1].z = 1.0f / p[1].z;
                            uvs[2].z = 1.0f / p[2].z;
                            */

                        }
                        else
                        {
                            uv = (uvs[0] * bc.x + uvs[1] * bc.y + uvs[2] * bc.z);
                        }

                        /*
                        if (perspectivecorrect)
                        {
                        float zz = 1.0f / (uvs[0].z * bc.x + uvs[1].z * bc.y + uvs[2].z * bc.z);
                        uv *= zz;
                        }
                        */
                        uv *= primitive.txHeight;

                        int cix = ((int)uv.y * primitive.txHeight + (int)uv.x) * 3;
                        cix = cix % (primitive.txHeight * primitive.txWidth * 3);


                        //primitive.dev_verticesOut[primitive.dev_indices[i]].texcoord0;
                        unsigned char tx1 = primitive.dev_diffuseTex[cix];
                        unsigned char tx2 = primitive.dev_diffuseTex[cix + 1];
                        unsigned char tx3 = primitive.dev_diffuseTex[cix + 2];

                        unsigned int red = tx1;
                        unsigned int green = tx2;
                        unsigned int blue = tx3;


                        //finalcolor = glm::vec3((float)red / 255.0, (float)green / 255.0, (float)blue / 255.0);
                        dev_fragBuffer[index].color = glm::vec3((float)red / 255.0, (float)green / 255.0, (float)blue / 255.0);
                        dev_fragBuffer[index].depth = depth;
                    }
                }
                else if (mode == 1 && (bc.x <= 0.01 || bc.y <= 0.01 || bc.z <= 0.01))
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyeNor };


                        glm::vec3 e[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyePos };

                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);

                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;

                        depth = tmp_depth;
                        dev_fragBuffer[index].color = glm::vec3(1.0, 0.0, 0.0);
                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;

                        float dt = glm::dot(glm::normalize(ee - pp), nn);
                        dev_fragBuffer[index].color *= dt;
                        dev_fragBuffer[index].depth = depth;

                    }
                }
                else if ((bc.x <= 0.02 && bc.y <= 0.02) ||
                         (bc.x <= 0.02 && bc.z <= 0.02) ||
                         (bc.y <= 0.02 && bc.z <= 0.02))
                {
                    // get the position from barycenter
                    glm::vec3 pp = p[0] * bc.x + p[1] * bc.y + p[2] * bc.z;

                    float tmp_depth = pp.z;
                    if (tmp_depth < depth)
                    {
                        glm::vec3 n[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyeNor,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyeNor };


                        glm::vec3 e[] = { (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 1]].eyePos,
                            (glm::vec3)primitive.dev_verticesOut[primitive.dev_indices[i + 2]].eyePos };

                        glm::vec3 nn = n[0] * bc.x + n[1] * bc.y + n[2] * bc.z;
                        nn = glm::normalize(nn);

                        glm::vec3 ee = e[0] * bc.x + e[1] * bc.y + e[2] * bc.z;


                        depth = tmp_depth;
                        dev_fragBuffer[index].color = glm::vec3(1.0, 0.0, 0.0);
                        dev_fragBuffer[index].eyeNor = nn;
                        dev_fragBuffer[index].eyePos = ee;

                        //float dt = glm::dot(glm::normalize(ee - pp), nn);
                        //dev_fragBuffer[index].color *= dt;
                        dev_fragBuffer[index].depth = depth;

                    }
                }
            }
            //printf("3");
        }
    }
}



//
// Perform rasterization.
//
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal,
               int displaymode, bool perepectivecorrect, bool spec, bool aa, bool supersample, 
               bool culling, bool testingmode, bool aabbcheck, bool cheapculling) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    // Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)



    cudaEvent_t start_vertexTransformAndAssembly, stop_vertexTransformAndAssembly;
    cudaEvent_t start_scanline, stop_scanline;
    cudaEvent_t start_render, stop_render;
    cudaEvent_t start_aa, stop_aa;
    cudaEvent_t start_downsample, stop_downsample;

    float ms_vertexTransformAndAssembly = 0.0f;
    float ms_scanline = 0.0f;
    float ms_render = 0.0f;
    float ms_aa = 0.0f;
    float ms_downsample = 0.0f;

    float ms1 = 0.0f;
    float ms2 = 0.0f;
    float ms3 = 0.0f;
    float ms4 = 0.0f;
    float ms5 = 0.0f;


    // update light transform
    glm::vec4 lightposition = MV * lightpos;
    lightposition /= lightposition.w;


    if (testingmode)
    {
        cudaEventCreate(&start_vertexTransformAndAssembly); 
        cudaEventCreate(&stop_vertexTransformAndAssembly); 
        cudaEventRecord(start_vertexTransformAndAssembly);
    }

    // Vertex Process & primitive assembly
    {
        curPrimitiveBeginId = 0;
        dim3 numThreadsPerBlock(32);

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

    if (testingmode)
    {
        cudaEventRecord(stop_vertexTransformAndAssembly); cudaEventSynchronize(stop_vertexTransformAndAssembly);
        ms1 = 0;
        cudaEventElapsedTime(&ms1, start_vertexTransformAndAssembly, stop_vertexTransformAndAssembly);
        ms_vertexTransformAndAssembly = ms1;
        cudaEventDestroy(start_vertexTransformAndAssembly);
        cudaEventDestroy(stop_vertexTransformAndAssembly);
    }

    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer);
    cudaMemset(dev_dsfragmentBuffer, 0, width * height * sizeof(Fragment));
    initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_dsfragmentBuffer);

    // TODO: rasterize
    auto it = mesh2PrimitivesMap.begin();
    auto itEnd = mesh2PrimitivesMap.end();

    //for (; it != itEnd; it++) {
    auto p = (it->second).begin();	// each primitive
    auto pEnd = (it->second).end();
    //for (; p != pEnd; ++p) {


    cudaMemcpy(dev_primitivestmp, dev_primitives, totalNumPrimitives*sizeof(Primitive), cudaMemcpyDeviceToDevice);
    int totalNumPrimitives_tmp = totalNumPrimitives;

    if (culling)
    {
        // remove backfaces -------------------------------------------------
        thrust::device_ptr <Primitive> thrust_prims(dev_primitivestmp);
        thrust::sort(thrust_prims, thrust_prims + totalNumPrimitives);

        //thrust::device_ptr<Primitive> thrust_prims2(dev_primitivestmp);
        thrust::device_ptr<Primitive> P = thrust::remove_if(thrust_prims, thrust_prims + totalNumPrimitives, is_backface());
        totalNumPrimitives_tmp = P - thrust_prims;
        // remove backfaces -------------------------------------------------
    }


    if (testingmode)
    {
        cudaEventCreate(&start_scanline);
        cudaEventCreate(&stop_scanline);
        cudaEventRecord(start_scanline);
    }

    //printf("totalNumPrimitives = %d\n", totalNumPrimitives);
    scanline << <blockCount2d, blockSize2d >> >(width, height,
                                                dev_fragmentBuffer,
                                                totalNumPrimitives_tmp, //p->numIndices,
                                                curPrimitiveBeginId,
                                                dev_primitivestmp,
                                                *p,
                                                displaymode,
                                                perepectivecorrect, 0, 0,
                                                aabbcheck, cheapculling);
    checkCUDAError("scanline");
    //accumframebuffers << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_dsfragmentBuffer);
    //checkCUDAError("accumframebuffers");
    
    if (testingmode)
    {
        cudaEventRecord(stop_scanline); cudaEventSynchronize(stop_scanline);
        ms2 = 0;
        cudaEventElapsedTime(&ms2, start_scanline, stop_scanline);
        ms_scanline = ms2;
        cudaEventDestroy(start_scanline);
        cudaEventDestroy(stop_scanline);
    }

    // antialias
    // shift
    if (aa && !supersample)
    {
        if (testingmode)
        {
            cudaEventCreate(&start_aa);
            cudaEventCreate(&stop_aa);
            cudaEventRecord(start_aa);
        }

        glm::vec3 black(0.0f);
        clearVertexBuffer << <blockCount2d, blockSize2d >> >(width * height, dev_dsfragmentBuffer, black);
        checkCUDAError("clearVertexBuffer");

        scanline << <blockCount2d, blockSize2d >> >(width, height,
                                                    dev_dsfragmentBuffer,
                                                    totalNumPrimitives_tmp, //p->numIndices,
                                                    curPrimitiveBeginId,
                                                    dev_primitivestmp,
                                                    *p,
                                                    displaymode,
                                                    perepectivecorrect, 0.2, 0,
                                                    aabbcheck, cheapculling);

        checkCUDAError("Scanline");
        accumframebuffers << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_dsfragmentBuffer);
        checkCUDAError("accumframebuffers");

        clearVertexBuffer << <blockCount2d, blockSize2d >> >(width * height, dev_dsfragmentBuffer, black);
        checkCUDAError("clearVertexBuffer");
        scanline << <blockCount2d, blockSize2d >> >(width, height,
                                                    dev_dsfragmentBuffer,
                                                    totalNumPrimitives_tmp, //p->numIndices,
                                                    curPrimitiveBeginId,
                                                    dev_primitivestmp,
                                                    *p,
                                                    displaymode,
                                                    perepectivecorrect, 0, 0.25,
                                                    aabbcheck, cheapculling);

        checkCUDAError("Scanline");
        accumframebuffers << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_dsfragmentBuffer);
        checkCUDAError("accumframebuffers");

        clearVertexBuffer << <blockCount2d, blockSize2d >> >(width * height, dev_dsfragmentBuffer, black);
        checkCUDAError("clearVertexBuffer");
        scanline << <blockCount2d, blockSize2d >> >(width, height,
                                                    dev_dsfragmentBuffer,
                                                    totalNumPrimitives_tmp, //p->numIndices,
                                                    curPrimitiveBeginId,
                                                    dev_primitivestmp,
                                                    *p,
                                                    displaymode,
                                                    perepectivecorrect, 0.25, 0.2,
                                                    aabbcheck, cheapculling);

        checkCUDAError("Scanline");
        accumframebuffers << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_dsfragmentBuffer);
        checkCUDAError("accumframebuffers");



        multframebuffer << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, 0.25f);
        checkCUDAError("mergeframebuffer");
        //}

        //curPrimitiveBeginId += p->numPrimitives;
        //}

        if (testingmode)
        {
            cudaEventRecord(stop_aa); cudaEventSynchronize(stop_aa);
            ms3 = 0;
            cudaEventElapsedTime(&ms3, start_aa, stop_aa);
            ms_aa = ms3;
            cudaEventDestroy(start_aa);
            cudaEventDestroy(stop_aa);
        }
    }


    if (testingmode)
    {
        cudaEventCreate(&start_render);
        cudaEventCreate(&stop_render);
        cudaEventRecord(start_render);
    }

    // Copy depthbuffer colors into framebuffer
    render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, lightposition, lightcol, spec);
    checkCUDAError("fragment shader");
    
    if (testingmode)
    {
        cudaEventRecord(stop_render); cudaEventSynchronize(stop_render);
        ms4 = 0;
        cudaEventElapsedTime(&ms4, start_render, stop_render);
        ms_render = ms4;
        cudaEventDestroy(start_render);
        cudaEventDestroy(stop_render);
    }


    if (supersample)
    {
        if (testingmode)
        {
            cudaEventCreate(&start_downsample);
            cudaEventCreate(&stop_downsample);
            cudaEventRecord(start_downsample);
        }

        downsample << <blockCount2d, blockSize2d >> >(cwidth, cheight, dev_framebuffer, dev_dsframebuffer, aa);
        checkCUDAError("downsample");

        if (testingmode)
        {
            cudaEventRecord(stop_downsample); cudaEventSynchronize(stop_downsample);
            ms5 = 0;
            cudaEventElapsedTime(&ms5, start_downsample, stop_downsample);
            ms_downsample = ms5;
            cudaEventDestroy(start_downsample);
            cudaEventDestroy(stop_downsample);
        }

        // Copy framebuffer into OpenGL buffer for OpenGL previewing
        sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, cwidth, cheight, dev_dsframebuffer);
        checkCUDAError("copy render result to pbo");
    }
    else
    {
        // Copy framebuffer into OpenGL buffer for OpenGL previewing
        sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
        checkCUDAError("copy render result to pbo");
    }

    if (testingmode)
    {
        printf("[%f, %f, %f, %f, %f],\n", ms_vertexTransformAndAssembly,
                                   ms_scanline,
                                   ms_aa,
                                   ms_render,
                                   ms_downsample);
    }
}

//
// Called once at the end of the program to free CUDA memory.
//
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

    cudaFree(dev_dsframebuffer);
    dev_dsframebuffer = NULL;

    cudaFree(dev_depth);
    dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
