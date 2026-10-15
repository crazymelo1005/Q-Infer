// --vision-gpu 的默认值与默认编码器二进制的选择逻辑（2026-10-15 翻转默认）：
// 默认 on（Intel iGPU 上的 ggml-vulkan 档，7.0x、零 VRAM、质量在管线噪声底内），
// 显式 --vision-gpu off 才回 CPU 档；默认二进制跟着档走（build-vision-vk/ vs build-vision/）。
#include "serve/server.hpp"

#include "check.hpp"

using namespace qinfer::serve;

int main() {
    // 1. 默认 on：不带 --vision-gpu 启动即选 iGPU 档。
    {
        ServeConfig cfg;
        CHECK(cfg.vision_gpu);
    }
    // 2. 默认二进制随档位：gpu 档取 build-vision-vk/，CPU 档取 build-vision/。
    {
        CHECK(std::string(default_vision_binary(true)) == "build-vision-vk/bin/qinfer-vision");
        CHECK(std::string(default_vision_binary(false)) == "build-vision/bin/qinfer-vision");
    }
    return 0;
}
