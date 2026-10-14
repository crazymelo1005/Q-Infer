// Media fetch: normalize data:/http(s)/file:// URLs to local temporary files.
// Used by the server to resolve image_url parts in chat requests to paths the vision
// encoder can read. Size-capped and media-root restricted for file:// URLs.
#pragma once

#include <string>

namespace qinfer::serve {

struct MediaResult {
    bool ok = false;
    std::string mime;   // detected or extracted from Content-Type / data: prefix
    std::string err;
};

// Fetch the media at `url` and write it to `out_path`.
// - data:[<mime>][;base64],<data> → decode and write
// - http(s):// → HTTP GET (32 MiB cap) → write
// - file://<path> → copy, but only if `media_root` is non-empty and `path` starts with it
// Returns ok=false with an err message on failure.
MediaResult fetch_media(const std::string& url, const std::string& out_path,
                        const std::string& media_root = "");

// Detect MIME type from file extension (lowercased). Returns application/octet-stream for
// unrecognized extensions.
std::string detect_mime(const std::string& path);

// Standard base64 decode. Returns empty string on invalid input (graceful degradation).
std::string base64_decode(const std::string& encoded);

}  // namespace qinfer::serve
