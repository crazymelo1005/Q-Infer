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

## Notes that matter on this iGPU

- **`precise` is load-bearing.** The Intel compiler contracts `a + b * c` into
  an FMA by default; without `precise` on the accumulators the shader's last
  bits diverge from the host (`L1 4e-08` but `bit-same 0.56`). The CUDA kernels
  get the same guarantee from `__fmul_rn`/`__fadd_rn`.
- The device has **no `VK_KHR_shader_integer_dot_product`** (and no fp16/int8
  extensions): integer dot products are hand-written, as the spec requires.
- fp16 scales are decoded by bit manipulation (`f16()` in the shader), matching
  `src/kernels/fp16.hpp` `f16_bits_to_f32`.