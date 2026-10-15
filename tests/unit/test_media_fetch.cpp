// Media fetch: data:/http(s)/file:// URL normalization to temporary files.
// Tests the pure functions (base64 decode, mime detection, path validation) without
// requiring network access or actual HTTP servers.
#include "serve/media_fetch.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"

namespace fs = std::filesystem;
using namespace qinfer::serve;

int main() {
    // ---- data: URL handling ----
    // Minimal 1x1 PNG (base64).
    {
        const std::string url =
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        const std::string out = "/tmp/test_media_fetch.png";
        auto result = fetch_media(url, out);
        CHECK(result.ok);
        CHECK(result.mime == "image/png");
        CHECK(fs::exists(out));
        CHECK(fs::file_size(out) > 0);
        fs::remove(out);
    }

    // data: with charset parameter (should still extract mime correctly).
    {
        const std::string url = "data:image/jpeg;base64,/9j/4AAQSkZJRg==";
        const std::string out = "/tmp/test_media_fetch.jpg";
        auto result = fetch_media(url, out);
        CHECK(result.ok);
        CHECK(result.mime == "image/jpeg");
        fs::remove(out);
    }

    // data: without explicit mime defaults to application/octet-stream.
    {
        const std::string url = "data:;base64,SGVsbG8=";
        const std::string out = "/tmp/test_media_fetch.bin";
        auto result = fetch_media(url, out);
        CHECK(result.ok);
        CHECK(result.mime == "application/octet-stream");
        fs::remove(out);
    }

    // ---- file:// URL handling ----
    // Create a test file.
    {
        const std::string test_dir = "/tmp/media_fetch_test";
        fs::create_directories(test_dir);
        const std::string test_file = test_dir + "/test.png";
        std::ofstream(test_file) << "fake png data";

        // file:// within media-root should succeed.
        auto result = fetch_media("file://" + test_file, "/tmp/test_out.png", test_dir);
        CHECK(result.ok);
        CHECK(fs::exists("/tmp/test_out.png"));
        fs::remove("/tmp/test_out.png");

        // file:// outside media-root should be rejected.
        auto bad_result = fetch_media("file:///etc/passwd", "/tmp/test_out", test_dir);
        CHECK(!bad_result.ok);
        CHECK(bad_result.err.find("outside media-root") != std::string::npos);

        fs::remove_all(test_dir);
    }

    // file:// with empty media-root should be rejected (no unrestricted file access).
    {
        auto result = fetch_media("file:///etc/passwd", "/tmp/test_out", "");
        CHECK(!result.ok);
        CHECK(result.err.find("media-root") != std::string::npos);
    }

    // ---- mime detection from extension ----
    CHECK(detect_mime("photo.jpg") == "image/jpeg");
    CHECK(detect_mime("photo.jpeg") == "image/jpeg");
    CHECK(detect_mime("photo.png") == "image/png");
    CHECK(detect_mime("photo.webp") == "image/webp");
    CHECK(detect_mime("photo.gif") == "image/gif");
    CHECK(detect_mime("unknown.xyz") == "application/octet-stream");

    // ---- base64 decode ----
    CHECK(base64_decode("SGVsbG8=") == "Hello");
    CHECK(base64_decode("V29ybGQ=") == "World");
    CHECK(base64_decode("") == "");
    // Invalid base64 returns empty (graceful degradation).
    CHECK(base64_decode("!!!invalid!!!").empty());

    return 0;
}
