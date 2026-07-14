cmake_minimum_required(VERSION 3.18)
project(transformer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(transformer_core STATIC
    src/kernels.cu
    src/layers.cu
)
target_include_directories(transformer_core PUBLIC include)
target_link_libraries(transformer_core PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(transformer_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(train_transformer_cuda src/main.cu)
target_link_libraries(train_transformer_cuda PRIVATE transformer_core)

add_executable(probe src/probe.cu)
target_link_libraries(probe PRIVATE transformer_core)

# Link the bbpe tokenizer's C++ core (../tokenizer) so --tokenizer bbpe works.
# It's a pure-C++, Python-free library, so we just compile its sources into a
# static lib and link. Turn OFF to build with only the byte-level tokenizer.
option(WITH_BBPE "Link the bbpe tokenizer for --tokenizer bbpe" ON)
if(WITH_BBPE)
    set(BBPE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer)
    if(EXISTS ${BBPE_DIR}/src/bpe_model.cpp)
        find_package(Threads REQUIRED)
        add_library(bbpe_core STATIC
            ${BBPE_DIR}/src/unicode_utils.cpp
            ${BBPE_DIR}/src/pretokenizer.cpp
            ${BBPE_DIR}/src/bpe_model.cpp
            ${BBPE_DIR}/src/bpe_trainer.cpp
            ${BBPE_DIR}/src/corpus_reader.cpp
        )
        target_include_directories(bbpe_core PUBLIC ${BBPE_DIR}/include)
        target_link_libraries(bbpe_core PUBLIC Threads::Threads)
        target_compile_features(bbpe_core PUBLIC cxx_std_17)
        target_link_libraries(train_transformer_cuda PRIVATE bbpe_core)
        target_compile_definitions(train_transformer_cuda PRIVATE WITH_BBPE=1)
    else()
        message(WARNING "WITH_BBPE=ON but ${BBPE_DIR}/src not found; building without bbpe support")
    endif()
endif()
