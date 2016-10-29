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
	float signedArea = calculateSignedArea(tri);
	if (fabs(signedArea) < EPSILON) return 0.0f;
	return fabs(calculateSignedArea(baryTri) / signedArea);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
	//if (beta < 0.0f)
	//	beta = 0.f;
	//if (gamma < 0.0f)
	//	gamma = 0.f;
	if (beta > 1.0f) return glm::vec3(0.0f, 1.0f, 0.0f);
	else if (gamma > 1.0f) return glm::vec3(0.0f, 0.0f, 1.0f);
	else if (beta + gamma >1.0f) return glm::vec3(0.0f, beta, gamma);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

__host__ __device__ static
glm::vec3 getTextureColor(unsigned char *pTex, glm::vec2 texcoord, int w, int h, int component) {
		int x = 0.5f + (w - 1.f) * texcoord.x;
		int y = 0.5f + (h - 1.f) * texcoord.y;
		float scale = 1.0f / 255.0f;
		int index = x + y * w;
		//if (index < 0 || index >= w * h)
		//	printf("%d %d\n",index,w*h);
		//if (index < 0) index = 0;
		//else if (index >= w * h) index = w * h - 1;

		return scale * glm::vec3(pTex[index * component],
			pTex[index * component + 1],
			pTex[index * component + 2]);
}

__host__ __device__ static
	glm::vec3 getBilinearTextureColor(unsigned char *pTex, glm::vec2 texcoord, int w, int h, int component) {
		float u = (w - 1.f) * texcoord.x;
		float v = (h - 1.f) * texcoord.y;
		int x = floor(u);
		int y = floor(v);
		float u_ratio = u - x;
		float v_ratio = v - y;
		float u_oppsite = 1 - u_ratio;
		float v_oppsite = 1 - v_ratio;
		int xNext = x + 1;
		int yNext = y + 1;
		if (xNext >= w)
			xNext = w - 1;
		if (yNext >= h)
			yNext = h - 1;
		int index0 = x + y * w;
		int index1 = xNext + y * w;
		int index2 = x + yNext * w;
		int index3 = xNext + yNext * w;

		glm::vec3 c0(pTex[index0 * component], pTex[index0 * component + 1], pTex[index0 * component + 2]);
		glm::vec3 c1(pTex[index1 * component], pTex[index1 * component + 1], pTex[index1 * component + 2]);
		glm::vec3 c2(pTex[index2 * component], pTex[index2 * component + 1], pTex[index2 * component + 2]);
		glm::vec3 c3(pTex[index3 * component], pTex[index3 * component + 1], pTex[index3 * component + 2]);

		float scale = 1.0f / 255.0f;
		//if (index < 0 || index >= w * h)
		//	printf("%d %d\n",index,w*h);
		//if (index < 0) index = 0;
		//else if (index >= w * h) index = w * h - 1;
		//return scale * c0;
		return scale * ((c0 * u_oppsite + c1 * u_ratio) * v_oppsite + (c2 * u_oppsite + c3 * u_ratio) * v_ratio);
}

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))