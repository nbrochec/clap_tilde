cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(clap_tilde VERSION 1.0.0)

set(CMAKE_OSX_ARCHITECTURES arm64)
include(FetchContent)

# ── libtorch ─────────────────────────────────────────────────────────────────
# Reuse the libtorch download from the sibling ipt_tilde project if present,
# otherwise fall back to downloading it here.
set(LIBTORCH_PYTHON_VERSION "3.12")
set(LIBTORCH_URL "https://anaconda.org/pytorch/pytorch/2.4.1/download/osx-arm64/pytorch-2.4.1-py3.12_0.tar.bz2")

set(IPT_LIBTORCH "${CMAKE_CURRENT_SOURCE_DIR}/../ipt_tilde/libs/libtorch")
set(LOCAL_LIBTORCH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch")

if(EXISTS "${IPT_LIBTORCH}/lib")
    message(STATUS "Reusing libtorch from ipt_tilde: ${IPT_LIBTORCH}")
    set(LIBTORCH_DIR "${IPT_LIBTORCH}")
elseif(EXISTS "${LOCAL_LIBTORCH}/lib")
    message(STATUS "libtorch found locally: ${LOCAL_LIBTORCH}")
    set(LIBTORCH_DIR "${LOCAL_LIBTORCH}")
else()
    message(STATUS "Downloading libtorch...")
    FetchContent_Declare(
        libtorch
        URL ${LIBTORCH_URL}
        SOURCE_DIR ${LOCAL_LIBTORCH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(libtorch)
    set(LIBTORCH_DIR "${LOCAL_LIBTORCH}")
endif()

list(APPEND CMAKE_PREFIX_PATH
    "${LIBTORCH_DIR}/lib/python${LIBTORCH_PYTHON_VERSION}/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── r8brain ───────────────────────────────────────────────────────────────────
# Reuse from ipt_tilde
set(IPT_R8BRAIN "${CMAKE_CURRENT_SOURCE_DIR}/../ipt_tilde/libs/r8brain")
if(EXISTS "${IPT_R8BRAIN}/CMakeLists.txt")
    message(STATUS "Reusing r8brain from ipt_tilde")
    add_subdirectory("${IPT_R8BRAIN}" "${CMAKE_BINARY_DIR}/r8brain")
else()
    add_subdirectory(libs/r8brain)
endif()

# ── min-api ───────────────────────────────────────────────────────────────────
set(IPT_MIN_API "${CMAKE_CURRENT_SOURCE_DIR}/../ipt_tilde/min-api")
if(EXISTS "${IPT_MIN_API}/script/min-package.cmake")
    message(STATUS "Reusing min-api from ipt_tilde")
    set(C74_MIN_API_DIR "${IPT_MIN_API}")
else()
    set(C74_MIN_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/min-api")
endif()

include(${C74_MIN_API_DIR}/script/min-package.cmake)
set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/externals")

# ── onnxruntime ──────────────────────────────────────────────────────────────
set(ORT_VERSION "1.20.1")
set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
set(LOCAL_ORT "${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime")

if(EXISTS "${LOCAL_ORT}/include/onnxruntime_cxx_api.h")
    message(STATUS "onnxruntime found locally: ${LOCAL_ORT}")
else()
    message(STATUS "Downloading onnxruntime ${ORT_VERSION}...")
    FetchContent_Declare(
        onnxruntime
        URL ${ORT_URL}
        SOURCE_DIR ${LOCAL_ORT}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
endif()

set(ORT_INCLUDE_DIR "${LOCAL_ORT}/include")
set(ORT_LIB        "${LOCAL_ORT}/lib/libonnxruntime.${ORT_VERSION}.dylib")

# ── targets ───────────────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(clap_tilde)

# ── C++ standard / deployment target ─────────────────────────────────────────
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)
message(STATUS "MacOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
