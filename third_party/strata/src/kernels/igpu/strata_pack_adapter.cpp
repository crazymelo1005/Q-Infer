// Strata pack format adapter implementation.
//
// See strata_pack_adapter.hpp for the layout documentation.
#include "strata/kernels/igpu/strata_pack_adapter.hpp"

namespace strata::kernels::igpu {

namespace {

// Layout constants (from Strata expert.hpp: H=2560, FF=640, QK=64).
constexpr int H  = 2560;
constexpr int FF = 640;
constexpr int QK = 64;

constexpr int ROW_GU = H * 2 / 8;    // 640 bytes of codes per gate/up row
constexpr int ROW_D  = FF * 2 / 8;   // 160 bytes of codes per down row
constexpr int SC_GU  = H / QK;       // 40 weight blocks per gate/up row
constexpr int SC_D   = FF / QK;      // 10 weight blocks per down row

// Region offsets within the blob.
constexpr int O_GU_CODES  = 0;
constexpr int O_D_CODES   = 2 * FF * ROW_GU;            // 819200
constexpr int O_GU_SCALES = O_D_CODES + H * ROW_D;      // 1228800
constexpr int O_D_SCALES  = O_GU_SCALES + 2 * FF * SC_GU * 2;  // 1331200

}  // namespace

StrataPackRow gu_row(const uint8_t* blob, int r, bool is_up) {
    // Gate/up codes region: two sequential FF-row blocks.
    //   Gate: blob[0 + r * ROW_GU]
    //   Up:   blob[FF * ROW_GU + r * ROW_GU]
    const int codes_offset = (is_up ? FF * ROW_GU : 0) + r * ROW_GU;

    // Gate/up scales region: two sequential FF-row blocks of SC_GU*2 bytes each.
    //   Gate: blob[O_GU_SCALES + r * SC_GU * 2]
    //   Up:   blob[O_GU_SCALES + FF * SC_GU * 2 + r * SC_GU * 2]
    const int scales_offset = O_GU_SCALES + (is_up ? FF * SC_GU * 2 : 0) + r * SC_GU * 2;

    return {blob + codes_offset, blob + scales_offset, SC_GU};
}

StrataPackRow down_row(const uint8_t* blob, int r) {
    // Down codes: blob[O_D_CODES + r * ROW_D]
    // Down scales: blob[O_D_SCALES + r * SC_D * 2]
    return {blob + O_D_CODES + r * ROW_D,
            blob + O_D_SCALES + r * SC_D * 2,
            SC_D};
}

}  // namespace strata::kernels::igpu
