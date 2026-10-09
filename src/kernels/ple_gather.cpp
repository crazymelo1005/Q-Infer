#include "kernels/ple_gather.hpp"

#include <cstring>

namespace qinfer::kernels {

void assemble_ple_vector(const float (*rows)[kPleHeadDim], float* out2560) {
    for (int h = 0; h < kPleNHeads; ++h) {
        std::memcpy(out2560 + h * kPleHeadDim, rows[h], sizeof(float) * kPleHeadDim);
    }
}

}  // namespace qinfer::kernels
