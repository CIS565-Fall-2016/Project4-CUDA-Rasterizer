/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
    AABB aabb;
    aabb.min = glm::vec3(
            min(min(tri[0].x, tri[1].x), tri[2].x),
            min(min(tri[0].y, tri[1].y), tri[2].y),
            min(min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            max(max(tri[0].x, tri[1].x), tri[2].x),
            max(max(tri[0].y, tri[1].y), tri[2].y),
            max(max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.f && barycentricCoord.x <= 1.f &&
           barycentricCoord.y >= 0.f && barycentricCoord.y <= 1.f &&
           barycentricCoord.z >= 0.f && barycentricCoord.z <= 1.f;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z;
}

 /**
  * For a given barycentric coordinate, compute the corresponding vec3
  * on the triangle.
  */
  __host__ __device__ static
  glm::vec3 getVec3AtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 input[3]) {
 	 return barycentricCoord.x * input[0]
 	 		+ barycentricCoord.y * input[1]
 			+ barycentricCoord.z * input[2];
  }

  /**
   * For a given barycentric coordinate, compute a float on the triangle.
  */
  __host__ __device__ static
  float getFloatAtCoordinate(const glm::vec3 barycentricCoord, const float input[3]) {
  	return barycentricCoord.x * input[0]
  		   + barycentricCoord.y * input[1]
  		   + barycentricCoord.z * input[2];
  }

/**
 * For a given barycentric coordinate, compute the corresponding vec2
 * on the triangle.
 */
__host__ __device__ static
glm::vec2 getVec2AtCoordinate(const glm::vec3 barycentricCoord, const glm::vec2 input[3]) {
	 return barycentricCoord.x * input[0]
	 		+ barycentricCoord.y * input[1]
			+ barycentricCoord.z * input[2];
}

/**
 * For a given texture data pointer, compute color at spcified texcoord.
*/
__host__ __device__ static
glm::vec3 getColorFromTextureAtCoordinate(const unsigned char *pTex,
		const glm::vec2 texcoord, int w, int h, int stride) {
	const int x = (int)((w - 1.f) * texcoord.x + .5f);
	const int y = (int)((h - 1.f) * texcoord.y + .5f);
	const float scale = 1.f / 255.f;
	const int index = x + y * w;

	return scale * glm::vec3(pTex[index * stride],
			pTex[index * stride + 1],
			pTex[index * stride + 2]);
}

/**
 * For a given texture data pointer, compute color at spcified texcoord.
*/
__host__ __device__ static
glm::vec3 getColorFromTextureAtCoordinateBilinear(const unsigned char *pTex,
		const glm::vec2 texcoord, int w, int h, int stride) {
	const float scale = 1.f / 255.f;
	const float x = (w - 1.f) * texcoord.x;
	const float y = (h - 1.f) * texcoord.y;
	const int xi = (int)x;
	const int yi = (int)y;
	const float ux = x - xi;
	const float uy = y - yi;
	glm::vec3 c00(0.f);
	glm::vec3 c01(0.f);
	glm::vec3 c10(0.f);
	glm::vec3 c11(0.f);

	{
		const int index = xi + yi * w;
		c00 = glm::vec3(pTex[index * stride],
				pTex[index * stride + 1],
				pTex[index * stride + 2]);
	}
	if (yi < h - 1) {
		const int index = xi + (yi + 1) * w;
		c01 = glm::vec3(pTex[index * stride],
				pTex[index * stride + 1],
				pTex[index * stride + 2]);
	}
	if (xi < w - 1) {
		const int index = (xi + 1) + yi * w;
		c10 = glm::vec3(pTex[index * stride],
				pTex[index * stride + 1],
				pTex[index * stride + 2]);
	}
	if (yi < h - 1 && xi < w - 1) {
		const int index = (xi + 1) + (yi + 1) * w;
		c11 = glm::vec3(pTex[index * stride],
				pTex[index * stride + 1],
				pTex[index * stride + 2]);
	}

	return scale * ((1.f - ux) * ((1.f - uy) * c00 + uy * c01)
			+ ux * ((1.f - uy) * c10 + uy * c11));
}

/**
 * For a given barycentric coordinate, compute the corresponding perspective
 * corrected texcoord on the triangle.
 */
__host__ __device__ static
glm::vec2 getPerspectiveCorrectedTexcoordAtCoordinate(const glm::vec3 baryCoord,
		const glm::vec2 _texcoord[3], const float triDepth_1[3]) {
	const glm::vec2 texcoord[3] = {
		_texcoord[0] * triDepth_1[0],
		_texcoord[1] * triDepth_1[1],
		_texcoord[2] * triDepth_1[2]
	};
	const glm::vec2 numerator = getVec2AtCoordinate(baryCoord, texcoord);
	const float denomenator = getFloatAtCoordinate(baryCoord, triDepth_1);

	return numerator / denomenator;
}

/**
 * Linear interpolate between 2 points.
 */
__host__ __device__ static
glm::vec3 getVec3AtU(const float u, const glm::vec3 &a, const glm::vec3 &b) {
	return (1.f - u) * a + u * b;
}
