

project(main)

cmake_minimum_required(VERSION 3.22)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

if($ENV{VERBOSE})
    MESSAGE(STATUS "Benchmark GEMM")
endif()


find_package(BLAS)
find_package(LAPACK)
find_package(hip)


if($ENV{VERBOSE})
	    MESSAGE(STATUS "GEMM -- BLAS_LIBRARIES   : ${BLAS_LIBRARIES}")
	    MESSAGE(STATUS "GEMM -- BLAS_LINKER_FLAGS   : ${BLAS_LINKER_FLAGS}")
	    MESSAGE(STATUS "GEMM -- LAPACK_LIBRARIES   : ${LAPACK_LIBRARIES}")
	    MESSAGE(STATUS "GEMM -- LAPACK_LINKER_FLAGS   : ${LAPACK_LINKER_FLAGS}")
endif()


MESSAGE("{COMPILE_WITH_HIP}")

include_directories(${HIP_PATH}/include)
include_directories(/opt/rocm-5.7.2/hip/include)
include_directories(/opt/rocm-5.7.2/hipblas/include)
include_directories(/opt/rocm-5.7.2/hipblas/lib)
#include_directories(/nvme0/lemoinep/soft/rocThrust/include)

include_directories(/opt/rocm-5.7.2/roctracer/include)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.7.2 /opt/rocm-5.7.2)

# Set the GPU to compile for
#set(GPU_TARGETS "gfx900:gfx906:gfx1032;gfx1035" CACHE STRING "GPU targets to compile for")

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA.")
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})

set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

set(ROCM_ROOT "/opt/rocm-5.7.2" CACHE PATH "Root directory of the ROCm installation")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")



if(FALSE)
    FIND_PACKAGE(rocthrust REQUIRED)
    if (rocthrust_FOUND)
        MESSAGE("{rocthrust_LIBRARIES}")
    endif(rocthrust_FOUND)
endif()


#BEGIN::rocTHrust
if(FALSE)
    file(GLOB_RECURSE rocThrust_list 
        "${CMAKE_CURRENT_SOURCE_DIR}/../soft/rocThrust/include/thrust/*.h"
    )
    if (FALSE)
        foreach(file_path ${rocThrust_list})
            message(${file_path})
        endforeach()
    endif()

    MACRO(HEADER_DIRECTORIES return_list)
        FILE(GLOB_RECURSE new_list ${CMAKE_CURRENT_SOURCE_DIR}/../soft/rocThrust/include/thrust/*.h)
        SET(dir_list "")
        FOREACH(file_path ${new_list})
            GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
            SET(dir_list ${dir_list} ${dir_path})
        ENDFOREACH()
        LIST(REMOVE_DUPLICATES dir_list)
        SET(${return_list} ${dir_list})
    ENDMACRO()

    HEADER_DIRECTORIES(header_rocThrust_dir_list)

    list(LENGTH header_rocThrust_dir_list header_dir_list_count)
    message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")

    if (TRUE)
        foreach(dir_path ${header_rocThrust_dir_list})
            message(${dir_path})
        endforeach()
    endif()
endif()
#END::rocTHrust


message(CURRENT DIR= [${CMAKE_CURRENT_SOURCE_DIR}])
message(CURRENT DIR PREFIX_PATH= [${MAKE_PREFIX_PATH}])


if(FALSE)
   find_package(rocthrust REQUIRED CONFIG PATHS "/nvme0/lemoinep/soft/rocThrust")
   if (rocthrust_FOUND)
      message("{rocthrust_LIBRARIES}") 
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/nvme0/lemoinep/soft/rocThrust/include")
      list(APPEND include_dirs "/nvme0/lemoinep/soft/rocThrust/include")
   else()
      message(FATAL_ERROR "Could not find rocThrust")     
   endif()
endif()


if(TRUE)
   find_package(rocthrust REQUIRED CONFIG PATHS "/nvme0/lemoinep/soft/rocThrust")
   if (rocthrust_FOUND)
      message("{rocthrust_LIBRARIES}") 
      set(CMAKE_PREFIX_PATH "/nvme0/lemoinep/soft/rocThrust/include")
      list(APPEND include_dirs "/nvme0/lemoinep/soft/rocThrust/include")
   else()
      message(FATAL_ERROR "Could not find rocThrust")     
   endif()
endif()




#if(NOT TARGET roc::rocthrust)
 #   find_dependency(rocprim)
 # add_library(roc::rocthrust INTERFACE IMPORTED)
  #set_target_properties(roc::rocthrust PROPERTIES
   # INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../soft/rocThrust/include"
    #INTERFACE_LINK_LIBRARIES "roc::rocprim"
  #)
#endif()

if(FALSE)
    find_package(GMSH REQUIRED)
    if (GMSH _FOUND)
      message("{GMSH_LIBRARIES}") 
   else()
      message(FATAL_ERROR "Could not find GMSH")     
   endif()
endif()


find_library(GMSH_LIB gmsh)
if(NOT GMSH_LIB)
  message(FATAL_ERROR "Could not find libgmsh")
endif()





file(GLOB SOURCES
    "*.cpp"
)

add_executable(main ${SOURCES})

target_link_libraries(main
    PUBLIC
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        roc::rocprim
        roc::rocthrust
        ${GMSH_LIB}
    PRIVATE hipblas
    PRIVATE /opt/rocm-5.7.2/lib/libroctx64.so
)


if(GPU_RUNTIME STREQUAL "HIP")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
    list(APPEND include_dirs "${CMAKE_CURRENT_SOURCE_DIR}/../soft/rocPRIM/include")
    list(APPEND include_dirs "${CMAKE_CURRENT_SOURCE_DIR}/../soft/rocThrust/include")
    MESSAGE(${ROCM_ROOT}/include)
endif()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})
#set_source_files_properties(main.cpp utils.h vec.h PROPERTIES LANGUAGE ${GPU_RUNTIME})

#target_include_directories(
    #main
    #PRIVATE
        #${header_specx_dir_list}
    #PUBLIC
    #    ${header_rocThrust_dir_list}
#)


