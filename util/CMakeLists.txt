set(SOURCE_FILES
    "utilityCore.hpp"
    "utilityCore.cpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "tiny_gltf_loader.h"
    "timer.h"
    "timer.cpp"
   )

cuda_add_library(util
   ${SOURCE_FILES}
   OPTIONS -arch=sm_50
   )
