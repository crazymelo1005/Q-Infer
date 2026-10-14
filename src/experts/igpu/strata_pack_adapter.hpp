// Strata pack format adapter.
//
// Strata's expert blob stores Q2_0 weights in a planar layout: all gate/up
// codes first, then all down codes, then all gate/up scales, then all down
// scales. This adapter provides a per-row view that extracts the codes pointer
// and scales pointer for a given row, so the iGPU shaders can read them from
// separate SSBOs.
//
// Layout constants (H=2560, FF=640, QK=64):
//   ROW_GU = H*2/8 = 640   bytes of codes per gate/up row
//   ROW_D  = FF*2/8 = 160  bytes of codes per down row
//   SC_GU  = H/QK = 40     weight blocks per gate/up row
//   SC_D   = FF/QK = 10    weight blocks per down row
//
//   O_GU_CODES  = 0          2*FF*ROW_GU = 819200 bytes
//   O_D_CODES   = 819200     1*H*ROW_D   = 409600 bytes
//   O_GU_SCALES = 1228800    2*FF*SC_GU*2 = 102400 bytes
//   O_D_SCALES  = 1331200    1*H*SC_D*2   = 51200 bytes
//   Total = 1382400 bytes
#pragma once

#include <cstdint>

namespace qinfer::experts::igpu {

struct StrataPackRow {
    const uint8_t* codes;   // ROW_GU bytes (640) for gate/up, ROW_D (160) for down
    const uint8_t* scales;  // n_blocks * 2 bytes (fp16 scales)
    int n_blocks;           // H/QK=40 for gate/up, FF/QK=10 for down
};

// Gate/up row r (r in [0, FF=640)). is_up=false for gate, true for up.
StrataPackRow gu_row(const uint8_t* blob, int r, bool is_up);

// Down row r (r in [0, H=2560)).
StrataPackRow down_row(const uint8_t* blob, int r);

}  // namespace qinfer::experts::igpu
