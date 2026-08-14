cmake_minimum_required(VERSION 3.14)
project(edmformer.cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml: use a local checkout when available (sibling directory by default),
# otherwise fetch it
set(EDMFORMER_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml" CACHE PATH "path to ggml source tree")
if(EXISTS "${EDMFORMER_GGML_DIR}/CMakeLists.txt")
    add_subdirectory(${EDMFORMER_GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)
else()
    include(FetchContent)
    FetchContent_Declare(ggml GIT_REPOSITORY https://github.com/ggml-org/ggml.git GIT_TAG master)
    FetchContent_MakeAvailable(ggml)
endif()

add_library(edmformer_core
    src/common.cpp
    src/compute.cpp
    src/mel.cpp
    src/ssl_conformer.cpp
    src/songformer.cpp
    src/postprocess.cpp
    src/pipeline.cpp
    src/audio.cpp
)
target_include_directories(edmformer_core PUBLIC src third_party)
target_link_libraries(edmformer_core PUBLIC ggml)

if(APPLE)
    target_link_libraries(edmformer_core PUBLIC "-framework AudioToolbox" "-framework CoreFoundation")
endif()

add_executable(edmformer-cli src/main.cpp)
target_link_libraries(edmformer-cli PRIVATE edmformer_core)

add_executable(edmformer-test tests/test_parity.cpp)
target_link_libraries(edmformer-test PRIVATE edmformer_core)
