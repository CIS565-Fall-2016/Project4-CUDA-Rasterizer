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

// For perspective correct linear interpolation
__host__ __device__ static
float getFloatAtCoordinate(const glm::vec3 barycentricCoord, float a0, float a1, float a2)
{
	return barycentricCoord.x * a0 + barycentricCoord.y * a1 + barycentricCoord.z * a2;
}

__host__ __device__ static
glm::vec3 getVec3AtCoordinate(const glm::vec3 &abc, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3)
{
	return abc.x * v1 + abc.y * v2 + abc.z * v3;
}

__host__ __device__ static
glm::vec2 getVec2AtCoordinate(const glm::vec3 &abc, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3)
{
	return abc.x * v1 + abc.y * v2 + abc.z * v3;
}

__host__ __device__ static
void projectPointOntoAxis(const glm::vec2 &axis, const glm::vec2 &p, float &mind, float &maxd)
{
	float d = glm::dot(axis, p);
	mind = d < mind ? d : mind;
	maxd = d > maxd ? d : maxd;
}

__host__ __device__ static
bool triAABBIntersect(const AABB &box, const glm::vec3 tri[3])
{
	glm::vec2 c((box.max.x + box.min.x) * .5f, (box.max.y + box.min.y) * .5f);
	float hx = (box.max.x - box.min.x) * .5f;
	float hy = (box.max.y - box.min.y) * .5f;
	glm::vec2 b0(-hx, -hy);
	glm::vec2 b1(hx, -hy);
	glm::vec2 b2(hx, hy);
	glm::vec2 b3(-hx, hy);
	glm::vec2 v0 = glm::vec2(tri[0]) - c;
	glm::vec2 v1 = glm::vec2(tri[1]) - c;
	glm::vec2 v2 = glm::vec2(tri[2]) - c;
	glm::vec2 e0 = v1 - v0;
	glm::vec2 e1 = v2 - v1;
	glm::vec2 e2 = v0 - v2;
	glm::vec2 axes[5] =
	{
		{ 1.f, 0.f },
		{ 0.f, 1.f },
		{ -e0.y, e0.x },
		{ -e1.y, e1.x },
		{ -e2.y, e2.x },
	};

	for (int i = 0; i < 5; ++i)
	{
		float boxMin = FLT_MAX, boxMax = -FLT_MAX;
		float triMin = FLT_MAX, triMax = -FLT_MAX;

		// project box onto the axis
		projectPointOntoAxis(axes[i], b0, boxMin, boxMax);
		projectPointOntoAxis(axes[i], b1, boxMin, boxMax);
		projectPointOntoAxis(axes[i], b2, boxMin, boxMax);
		projectPointOntoAxis(axes[i], b3, boxMin, boxMax);

		// project triangle onto the axis
		projectPointOntoAxis(axes[i], v0, triMin, triMax);
		projectPointOntoAxis(axes[i], v1, triMin, triMax);
		projectPointOntoAxis(axes[i], v2, triMin, triMax);

		bool noOverlap = triMin > boxMax || triMax < boxMin;
		if (noOverlap) // a separation axis has been found == no intersection
		{
			return false;
		}
	}

	return true;
}

__host__ __device__ bool isFrontFacing(const glm::vec3 tri[3], bool ccwIsFront = true)
{
	float z = (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y) - (tri[1].y - tri[0].y) * (tri[2].x - tri[0].x);
	if (ccwIsFront)
	{
		return z > 0.f;
	}
	else
	{
		return z < 0.f;
	}
}