cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(sog VERSION 1.0)

# Conditional path settings for Windows
if(WIN32)
    # Example Windows-specific path for libtorch
    list(APPEND CMAKE_PREFIX_PATH "C:/libtorch" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
else()
    set(CMAKE_PREFIX_PATH "/usr/local/include/libtorch" "/home/marosc/libtorch")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add AddressSanitizer flags
# if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
#     set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
# endif()

include_directories(utils)

# Conditional compilation options based on the compiler
if(MSVC)
    # MSVC-specific warnings and options
    add_compile_options(/W4 /permissive- /Ox)
else()
    # GCC/Clang-specific warnings and options
    add_compile_options(-Wall -Wextra -pedantic -Wshadow -Wfloat-equal -O3)
endif()

add_executable(sog main.cpp
                    src/utils/publicBeliefState.cpp
                    src/utils/card.cpp
                    src/utils/utils.cpp
                    src/tree/treeBuilder.cpp
                    src/solving/pbscfr.cpp
                    src/solving/deepResolving.cpp
                    src/solving/continualResolving.cpp
                    src/tree/exploitability.cpp
                    src/tree/resolvingStrategyFilling.cpp
                    src/nn/inference.cpp
                    src/nn/situationSolver.cpp
                    src/nn/net.cpp
                    src/nn/trainer.cpp
                    src/nn/dataSaver.cpp
                    src/nn/dataGenerator.cpp
                    src/nn/netHelper.cpp
                    src/game/equityCache.cpp
                    src/game/equity.cpp
                    src/game/dealer.cpp
                    src/game/gameState.cpp
                    src/settings.cpp
                    src/netHolder.cpp
                    src/game/handComp.cpp
                    src/tree/treeutils.cpp)

target_link_libraries(sog "${TORCH_LIBRARIES}")
set_property(TARGET sog PROPERTY CXX_STANDARD 20)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sog
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:sog>)
endif (MSVC)