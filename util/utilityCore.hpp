/**
 * @file      utilityCore.hpp
 * @brief     UTILITYCORE: A collection/kitchen sink of generally useful functions
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#pragma once

#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <fstream>

#define PI                          3.1415926535897932384626422832795028841971
#define TWO_PI                      6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD           0.5773502691896257645091487805019574556476
#define E                           2.7182818284590452353602874713526624977572
#define G                           6.67384e-11
#define EPSILON                     .000000001
#define ZERO_ABSORPTION_EPSILON     0.00001
#define PROFILING                   0
#define PROFILING_PREFIX            "C:\\Users\\213re\\Code\\coursework\\565CIS\\Project4-CUDA-Rasterizer\\profiling\\"

#ifndef PROFILE_KERNEL
#define PROFILE_KERNEL
#if PROFILING

#define START_PROFILE(name) \
  cudaEvent_t start_##name, stop_##name; \
  cudaEventCreate(&start_##name); \
  cudaEventCreate(&stop_##name); \
  cudaEventRecord(start_##name);

#define END_PROFILE(name) \
  cudaEventRecord(stop_##name); \
  cudaEventSynchronize(stop_##name); \
  float milliseconds_##name = 0; \
  cudaEventElapsedTime(&milliseconds_##name, start_##name, stop_##name); \
  cudaEventDestroy(start_##name); \
  cudaEventDestroy(stop_##name); \
  std::ofstream out_##name; \
  out_##name.open(PROFILING_PREFIX "PROFILE_" #name ".txt", std::ios::out | std::ios::app); \
  out_##name << milliseconds_##name << std::endl; \
  out_##name.close();

#else 

#define START_PROFILE(name)
#define END_PROFILE(name)

#endif
#endif

namespace utilityCore {
extern float clamp(float f, float min, float max);
extern bool replaceString(std::string &str, const std::string &from, const std::string &to);
extern glm::vec3 clampRGB(glm::vec3 color);
extern bool epsilonCheck(float a, float b);
extern std::vector<std::string> tokenizeString(std::string str);
extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
extern std::string convertIntToString(int number);
extern std::istream &safeGetline(std::istream &is, std::string &t); //Thanks to http://stackoverflow.com/a/6089413

//-----------------------------
//-------GLM Printers----------
//-----------------------------
extern void printMat4(const glm::mat4 &);
extern void printVec4(const glm::vec4 &);
extern void printVec3(const glm::vec3 &);
}