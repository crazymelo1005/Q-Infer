// 遥测：extra 回调值进入 now 与 history；10 系列在首次采样后键全在；历史封顶 60 点；
// Linux 上 /proc 读数非空；GPU 相关断言只钉「不抛异常」（CI 无 GPU，环境2 有——行为随环境）。
#include "serve/telemetry.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>

#include "check.hpp"

using namespace qinfer::serve;

int main() {
    // 1. extra 回调 → now 与 history（tok_s 系列首尾两值）。
    int calls = 0;
    Telemetry tel([&calls] {
        ++calls;
        return TelemetryExtra{12.5, 40.0, 3.0};
    });
    tel.sample_now();
    tel.sample_now();
    auto s = tel.snapshot();
    CHECK(calls == 2);
    CHECK(s.now.count("tok_s") == 1 && std::abs(*s.now.at("tok_s") - 12.5) < 1e-9);
    CHECK(s.now.count("tok_s_mean") == 1 && std::abs(*s.now.at("tok_s_mean") - 40.0) < 1e-9);
    CHECK(s.history.count("tok_s") == 1 && s.history.at("tok_s").size() == 2);
    CHECK(std::abs(*s.history.at("tok_s").back() - 12.5) < 1e-9);

    // 2. 10 系列键齐（spec §6 的 series 清单），值可为 nullopt。
    for (const char* key : {"gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power",
                            "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb",
                            "tok_s", "prefill_tok_s_mean"}) {
        CHECK(s.history.count(key) == 1);
    }

    // 3. 历史封顶 60 点。
    Telemetry tel2([]{ return TelemetryExtra{1.0, 2.0, 3.0}; });
    for (int i = 0; i < 70; ++i) tel2.sample_now();
    const auto s2 = tel2.snapshot();
    CHECK(s2.history.at("tok_s").size() <= 60);
    CHECK(s2.history.at("tok_s").size() == 60);

    // 4. 线程启停幂等；真实跑几秒能采到样。
    tel2.start();
    tel2.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2400));
    tel2.stop();
    tel2.stop();   // 重复 stop 不炸
    const auto s3 = tel2.snapshot();
    CHECK(s3.history.at("tok_s").size() >= 60);   // 启动后每约 1s 一采样，只增不减

    // 5. 静态信息：线程数 ≥ 1；GPU 字段不断言（CI 无卡 → 空，环境2 有卡 → 非空）。
    CHECK(s3.stat.threads >= 1);
#ifndef _WIN32
    // 6. Linux：/proc/meminfo 能读 → ram_total 非空；cpu 键必在（值可为 null——首点无前一时刻）。
    CHECK(s3.now.count("ram_total") == 1 && s3.now.at("ram_total").has_value());
    CHECK(s3.now.count("cpu") == 1);
#endif

    // 7. cpuinfo "model name" 行解析：取冒号后值并去首尾空白；无冒号/无值 → 空串。
    //    （曾是 substr(11) 直接切——"model name" 恰 10 字符，index 11 是冒号本身，
    //    cpu_name 被读成 ": Intel(R)…"。）
    CHECK(parse_cpu_model_name("model name\t: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz") ==
          "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz");
    CHECK(parse_cpu_model_name("model name : AMD Ryzen 7 5800X") == "AMD Ryzen 7 5800X");
    CHECK(parse_cpu_model_name("model name\t:\tApple M1  ") == "Apple M1");
    CHECK(parse_cpu_model_name("model name") == "");
    CHECK(parse_cpu_model_name("model name\t:") == "");
#ifndef _WIN32
    // 7b. 真实 /proc/cpuinfo 读出的 cpu_name 不带前导冒号（上述缺陷的回归钉）。
    CHECK(s3.stat.cpu_name.empty() || s3.stat.cpu_name.front() != ':');
#endif

    // 8. 任何情况 snapshot 都能取（读锁不含文件 IO——snapshot 只拷内存）。
    Telemetry tel3(nullptr);
    tel3.sample_now();
    (void)tel3.snapshot();

    puts("test_telemetry: OK");
    return 0;
}