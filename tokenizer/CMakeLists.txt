cmake_minimum_required(VERSION 3.16)
project(bbpe_simple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -DNDEBUG")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(Threads REQUIRED)

add_library(bbpe_core STATIC
    src/unicode_utils.cpp
    src/pretokenizer.cpp
    src/bpe_model.cpp
    src/bpe_trainer.cpp
    src/corpus_reader.cpp
)
target_include_directories(bbpe_core PUBLIC include)
target_link_libraries(bbpe_core PUBLIC Threads::Threads)

# stdc++fs wird von GCC < 9 für std::filesystem benötigt.
# Bei neueren Compilern ist das Flag harmlos (leere/kompatible Bibliothek).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(bbpe_core PUBLIC stdc++fs)
endif()

add_executable(train_tokenizer tools/train_tokenizer.cpp)
target_link_libraries(train_tokenizer PRIVATE bbpe_core)

add_executable(test_tokenizer tools/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE bbpe_core)
add_executable(encode_cli tools/encode_cli.cpp)
target_link_libraries(encode_cli PRIVATE bbpe_core)