# iGPU compute shaders (stage C)

GLSL sources compiled once into `.spv`, which are then embedded into
`spv_data.hpp` (both committed). The build never needs `glslangValidator` and
the binary needs no shader files at run time.

## Regenerating after editing a `.comp`

```sh
cd src/experts/igpu/shaders
glslangValidator -V q2_0_gemv.comp -o q2_0_gemv.spv
python3 embed.py            # rewrites spv_data.hpp from the .spv files
```

Then rebuild and run `test_igpu_gemv`. Commit the `.comp`, the `.spv` and
`spv_data.hpp` together.

## GEMV buffer contract

| Shader | Storage buffers, binding 0.. | Push constants |
| --- | --- | --- |
| `q2_0_gemv.comp` | weights (Q2_0), Q8_0 activation, output | rows, blocks |
| `iq2s_gemv.comp` | weights (IQ2_S), Q8_K activation, output, grid codebook (1024 x 8 B) | rows, blocks |
| `iq2xxs_gemv.comp` | weights (IQ2_XXS), Q8_K activation, output, grid codebook (256 x 8 B), sign patterns (128 B) | rows, blocks |
| `iq1m_gemv.comp` | weights (IQ1_M), Q8_K activation, output, grid codebook (2048 x 8 B, signed) | rows, blocks |

The IQ-format tables are static host data (`src/kernels/*_tables.hpp`), uploaded as plain storage
buffers rather than hardcoded: the SPIR-V stays small, and the numbers stay single-sourced with the
host kernels.

## Notes that matter on this iGPU

- **`precise` is load-bearing.** The Intel compiler contracts `a + b * c` into
  an FMA by default; without `precise` on the accumulators the shader's last
  bits diverge from the host (`L1 4e-08` but `bit-same 0.56`). The CUDA kernels
  get the same guarantee from `__fmul_rn`/`__fadd_rn`.
- The device has **no `VK_KHR_shader_integer_dot_product`** (and no fp16/int8
  extensions): integer dot products are hand-written, as the spec requires.
- fp16 scales are decoded by bit manipulation (`f16()` in the shader), matching
  `src/kernels/fp16.hpp` `f16_bits_to_f32`.