set(SOURCE_FILES
    "rasterize.cu"
	"common.h"
    "rasterize.h"
    "rasterizeTools.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
