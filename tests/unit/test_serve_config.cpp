// --vision-gpu 的默认值与默认编码器二进制的选择逻辑（2026-10-15 翻转默认）：
// 默认 on（Intel iGPU 上的 ggml-vulkan 档，7.0x、零 VRAM、质量在管线噪声底内），
// 显式 --vision-gpu off 才回 CPU 档；默认二进制跟着档走（build-vision-vk/ vs build-vision/）。
//
// 2026-10-16 追加的两个默认：专家档位缺省 = --resident-experts（命中走 DMA 而不是
// 页缓存 fault+memcpy，实测 4K 1.59K / 21K 2.37K / 131K 3.32K）；多模态缺省开，
// mmproj 从 --model 同目录的 mmproj*.gguf 里找，设备仍是 iGPU。
#include "serve/server.hpp"

#include "check.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer::serve;

namespace {

std::vector<std::string> args_with(std::initializer_list<const char*> extra) {
    std::vector<std::string> v{"--backend", "strata", "--engine-exe", "/x/strata", "--model", "/x/m.gguf"};
    for (const char* s : extra) v.emplace_back(s);
    return v;
}

}  // namespace

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
    // 3. 专家档位缺省：没点名任何一种就补 --resident-experts。
    {
        std::vector<std::string> a = args_with({});
        apply_expert_tier_default("strata", a);
        CHECK(a.back() == "--resident-experts");
        CHECK(!has_expert_tier(args_with({})));
    }
    // 4. 点名了就尊重：--mmap-experts / --resident-cpu-experts 都不再追加。
    for (const char* named : {"--mmap-experts", "--resident-cpu-experts", "--resident-experts"}) {
        std::vector<std::string> a = args_with({named});
        apply_expert_tier_default("strata", a);
        CHECK(has_expert_tier(a));
        int n = 0;
        for (const std::string& s : a) n += s == "--resident-experts" || s == "--mmap-experts" ||
                                             s == "--resident-cpu-experts";
        CHECK(n == 1);   // exactly the one the operator named
    }
    // 5. 非 strata 后端不动 engine_args（那里不允许 --engine-arg）。
    {
        std::vector<std::string> a = args_with({});
        apply_expert_tier_default("inhouse", a);
        CHECK(a.size() == 6 && a.back() == "/x/m.gguf");
    }
    // 6. --vision 三种取值：auto（默认）找得到就开、找不到就不开；on 找不到报错；off 永不启用。
    {
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / "qinfer-m5b-test";
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        const std::string model = (dir / "model.gguf").string();
        std::string mmproj, err;
        CHECK(resolve_vision(VisionRequest::Auto, model, "", mmproj, err) && mmproj.empty());   // none beside it
        CHECK(resolve_vision(VisionRequest::On, model, "", mmproj, err) == false && !err.empty());
        CHECK(resolve_vision(VisionRequest::Off, model, "", mmproj, err) && mmproj.empty());
        CHECK(resolve_vision(VisionRequest::Off, model, "/x/mmproj.gguf", mmproj, err) == false && !err.empty());
        CHECK(resolve_vision(VisionRequest::Auto, model, "/x/explicit.gguf", mmproj, err) &&
              mmproj == "/x/explicit.gguf");
        std::ofstream((dir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").string(), std::ios::binary) << "x";
        CHECK(resolve_vision(VisionRequest::Auto, model, "", mmproj, err) &&
              mmproj == (dir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").string());
        CHECK(resolve_vision(VisionRequest::On, model, "", mmproj, err) == true && !mmproj.empty());
        // a second one: still deterministic (lexicographically first), never ambiguous-and-silent
        std::ofstream((dir / "mmproj-zzz.gguf").string(), std::ios::binary) << "x";
        CHECK(find_mmproj_beside(model) == (dir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").string());
        // a same-prefix file that is not an mmproj gguf is ignored
        std::ofstream((dir / "mmproj-notes.txt").string(), std::ios::binary) << "x";
        CHECK(find_mmproj_beside(model).find(".gguf") != std::string::npos);
        std::filesystem::remove_all(dir, ec);
    }
    // 7. 模型目录不存在时也不能炸（find_* 返回空、auto 静默关掉）。
    {
        std::string mmproj, err;
        CHECK(resolve_vision(VisionRequest::Auto, "/no/such/dir/m.gguf", "", mmproj, err) && mmproj.empty());
        CHECK(resolve_vision(VisionRequest::On, "/no/such/dir/m.gguf", "", mmproj, err) == false);
    }
    return 0;
}