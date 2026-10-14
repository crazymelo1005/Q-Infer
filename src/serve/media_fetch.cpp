#include "serve/media_fetch.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace qinfer::serve {

namespace {

// RFC 4648 base64 alphabet. Returns empty string on invalid input.
std::string b64_decode(const std::string& in) {
    // Build reverse lookup table at first call.
    static int rev[256];
    static bool initialized = false;
    if (!initialized) {
        std::memset(rev, -1, sizeof(rev));
        const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) rev[static_cast<unsigned char>(table[i])] = i;
        initialized = true;
    }

    std::string out;
    out.reserve(in.size() * 3 / 4);
    int buf = 0, bits = 0;
    for (unsigned char c : in) {
        if (c == '=' || std::isspace(c)) continue;
        if (rev[c] < 0) return "";  // invalid character
        buf = (buf << 6) | rev[c];
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

std::string base64_decode(const std::string& encoded) {
    return b64_decode(encoded);
}

std::string detect_mime(const std::string& path) {
    const std::string ext = to_lower(fs::path(path).extension().string());
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".png") return "image/png";
    if (ext == ".webp") return "image/webp";
    if (ext == ".gif") return "image/gif";
    if (ext == ".bmp") return "image/bmp";
    return "application/octet-stream";
}

MediaResult fetch_media(const std::string& url, const std::string& out_path,
                        const std::string& media_root) {
    MediaResult r;

    if (url.rfind("data:", 0) == 0) {
        // data:[<mime>][;base64],<data>
        const std::string body = url.substr(5);
        const auto comma = body.find(',');
        if (comma == std::string::npos) {
            r.err = "data: URL missing comma";
            return r;
        }
        const std::string header = body.substr(0, comma);
        const std::string data = body.substr(comma + 1);

        // Extract mime (before the first semicolon, or the whole header if no semicolon).
        const auto semi = header.find(';');
        r.mime = semi == std::string::npos ? header : header.substr(0, semi);
        if (r.mime.empty()) r.mime = "application/octet-stream";

        // Check for ;base64 marker.
        const bool is_b64 = header.find(";base64") != std::string::npos;
        std::string decoded;
        if (is_b64) {
            decoded = b64_decode(data);
            if (decoded.empty() && !data.empty()) {
                r.err = "data: base64 decode failed";
                return r;
            }
        } else {
            // URL-encoded text (not commonly used for images, but handle it).
            decoded = data;
        }

        std::ofstream ofs(out_path, std::ios::binary);
        if (!ofs) {
            r.err = "cannot write to " + out_path;
            return r;
        }
        ofs.write(decoded.data(), static_cast<std::streamsize>(decoded.size()));
        r.ok = true;
        return r;
    }

    if (url.rfind("file://", 0) == 0) {
        // file://<path> — must be within media_root.
        if (media_root.empty()) {
            r.err = "file:// URLs require a media-root to be configured";
            return r;
        }
        const std::string path = url.substr(7);
        // Resolve both to absolute paths and check prefix.
        std::error_code ec;
        const fs::path resolved = fs::canonical(path, ec);
        if (ec) {
            r.err = "file:// path does not exist: " + path;
            return r;
        }
        const fs::path root = fs::canonical(media_root, ec);
        if (ec) {
            r.err = "media-root does not exist: " + media_root;
            return r;
        }
        const std::string resolved_str = resolved.string();
        const std::string root_str = root.string();
        if (resolved_str.rfind(root_str, 0) != 0) {
            r.err = "file:// path is outside media-root";
            return r;
        }
        // Copy the file.
        fs::copy_file(resolved, out_path, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            r.err = "failed to copy file: " + ec.message();
            return r;
        }
        r.mime = detect_mime(path);
        r.ok = true;
        return r;
    }

    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        // HTTP(S) fetch would go here. For now, return an error since we don't want to
        // pull in httplib for this module (it's in the server layer). The server will
        // handle HTTP fetches itself using its httplib client.
        r.err = "http(s):// fetch not implemented in media_fetch (server handles it)";
        return r;
    }

    r.err = "unsupported URL scheme: " + url.substr(0, url.find(':'));
    return r;
}

}  // namespace qinfer::serve
