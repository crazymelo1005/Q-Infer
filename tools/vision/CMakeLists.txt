# tools/vision - qinfer-vision, the encoder half of Q-Infer's multimodal path.  Ported from Strata's
# tools/vision (MIT, (c) 2026 Niko1221 and the Strata contributors; see NOTICE); it uses llama.cpp's mtmd
# library at the same pinned commit as the engine's ggml.
#
#   ../build_vision.sh                            # materializes the pinned llama.cpp, configures, builds
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=build-vision/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#   cmake --build build-vision --target qinfer-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  The option is kept in CMakeCache.txt, so a build folder configured before this default still
# needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(qinfer_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build-vision/llama.cpp" CACHE PATH "pinned llama.cpp tree, materialized by tools/build_vision.sh")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "qinfer-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "qinfer-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
# STRATA_VISION_VULKAN: run the encoder on the Intel iGPU through ggml-vulkan. The point of this
# arm is that the iGPU is UMA -- it holds the encoder's weights in system memory and costs the
# engine ZERO VRAM, which is exactly why STRATA_VISION_CUDA stays OFF (0.4-0.7 GB, upstream). The
# encoder is dense, batched and BF16, so it does not hit the "no integer dot product" wall that
# made the iGPU lose on the engine's quantized experts (ADR-019/020). Default OFF: measure first.
option(STRATA_VISION_VULKAN "run the vision encoder on the iGPU (builds ggml-vulkan)" OFF)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(qinfer-vision qinfer_vision.cpp)
target_link_libraries(qinfer-vision PRIVATE mtmd llama)
target_compile_features(qinfer-vision PRIVATE cxx_std_17)
set_target_properties(qinfer-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
