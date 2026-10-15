// Task 1: Strata pack format adapter unit test.
//
// Verifies that gu_row() and down_row() return the correct codes pointer,
// scales pointer, and n_blocks for each row of the planar Strata expert blob.
// The blob layout is:
//   [0..819200)       gate/up codes  (2 * FF * ROW_GU = 2*640*640)
//   [819200..1228800) down codes      (1 * H  * ROW_D  = 2560*160)
//   [1228800..1331200) gate/up scales (2 * FF * SC_GU * 2 = 2*640*40*2)
//   [1331200..1382400) down scales    (1 * H  * SC_D  * 2 = 2560*10*2)
#include "experts/igpu/strata_pack_adapter.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

using namespace qinfer::experts::igpu;

namespace {

// Layout constants (must match the adapter implementation).
constexpr int H  = 2560;
constexpr int FF = 640;
constexpr int QK = 64;

constexpr int ROW_GU = H * 2 / 8;    // 640 bytes per gate/up row of codes
constexpr int ROW_D  = FF * 2 / 8;   // 160 bytes per down row of codes
constexpr int SC_GU  = H / QK;       // 40 weight blocks per gate/up row
constexpr int SC_D   = FF / QK;      // 10 weight blocks per down row

constexpr int O_GU_CODES  = 0;
constexpr int O_D_CODES   = 819200;   // 2 * FF * ROW_GU
constexpr int O_GU_SCALES = 1228800;  // O_D_CODES + H * ROW_D
constexpr int O_D_SCALES  = 1331200;  // O_GU_SCALES + 2 * FF * SC_GU * 2
constexpr int TOTAL      = 1382400;   // O_D_SCALES + H * SC_D * 2

}  // namespace

int main() {
    // Allocate a blob of exactly the Strata expert size, filled with a known
    // pattern so we can verify pointer arithmetic by reading back values.
    std::vector<uint8_t> blob(TOTAL);
    for (int i = 0; i < TOTAL; ++i) {
        blob[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // --- Test 1: gu_row(blob, 0, false) = gate row 0 ---
    {
        StrataPackRow row = gu_row(blob.data(), 0, /*is_up=*/false);
        // Gate codes: blob[0 + 0 * ROW_GU] = blob.data() + 0
        CHECK(row.codes == blob.data() + O_GU_CODES);
        // Gate scales: blob[O_GU_SCALES + 0 * SC_GU * 2] = blob.data() + 1228800
        CHECK(row.scales == blob.data() + O_GU_SCALES);
        CHECK(row.n_blocks == SC_GU);  // 40
    }

    // --- Test 2: gu_row(blob, 0, true) = up row 0 ---
    {
        StrataPackRow row = gu_row(blob.data(), 0, /*is_up=*/true);
        // Up codes: blob[FF * ROW_GU + 0 * ROW_GU] = blob.data() + 409600
        CHECK(row.codes == blob.data() + FF * ROW_GU);
        // Up scales: blob[O_GU_SCALES + FF * SC_GU * 2 + 0] = blob.data() + 1228800 + 51200
        CHECK(row.scales == blob.data() + O_GU_SCALES + FF * SC_GU * 2);
        CHECK(row.n_blocks == SC_GU);  // 40
    }

    // --- Test 3: down_row(blob, 0) = down row 0 ---
    {
        StrataPackRow row = down_row(blob.data(), 0);
        // Down codes: blob[O_D_CODES + 0 * ROW_D] = blob.data() + 819200
        CHECK(row.codes == blob.data() + O_D_CODES);
        // Down scales: blob[O_D_SCALES + 0 * SC_D * 2] = blob.data() + 1331200
        CHECK(row.scales == blob.data() + O_D_SCALES);
        CHECK(row.n_blocks == SC_D);  // 10
    }

    // --- Test 4: Scale values can be read back correctly ---
    // Write a known fp16 value into the gate scales for row 5, block 3,
    // then read it through the adapter.
    {
        const int r = 5;
        const int b = 3;
        // Gate scales offset: O_GU_SCALES + r * SC_GU * 2 + b * 2
        const int scale_offset = O_GU_SCALES + r * SC_GU * 2 + b * 2;
        // Write a known fp16 value (0x3C00 = 1.0 in fp16).
        uint16_t known_fp16 = 0x3C00;
        std::memcpy(blob.data() + scale_offset, &known_fp16, 2);

        StrataPackRow row = gu_row(blob.data(), r, /*is_up=*/false);
        // The scales pointer should point to the start of row r's scales.
        const uint16_t* scales_ptr = reinterpret_cast<const uint16_t*>(row.scales);
        uint16_t read_back = scales_ptr[b];
        CHECK(read_back == known_fp16);
    }

    // --- Test 5: Non-zero row indices for gate/up ---
    {
        const int r = 100;
        StrataPackRow gate = gu_row(blob.data(), r, /*is_up=*/false);
        CHECK(gate.codes == blob.data() + O_GU_CODES + r * ROW_GU);
        CHECK(gate.scales == blob.data() + O_GU_SCALES + r * SC_GU * 2);

        StrataPackRow up = gu_row(blob.data(), r, /*is_up=*/true);
        CHECK(up.codes == blob.data() + O_GU_CODES + FF * ROW_GU + r * ROW_GU);
        CHECK(up.scales == blob.data() + O_GU_SCALES + FF * SC_GU * 2 + r * SC_GU * 2);
    }

    // --- Test 6: Non-zero row index for down ---
    {
        const int r = 2000;
        StrataPackRow row = down_row(blob.data(), r);
        CHECK(row.codes == blob.data() + O_D_CODES + r * ROW_D);
        CHECK(row.scales == blob.data() + O_D_SCALES + r * SC_D * 2);
        CHECK(row.n_blocks == SC_D);
    }

    std::printf("strata_pack_adapter: all tests passed\n");
    return 0;
}
