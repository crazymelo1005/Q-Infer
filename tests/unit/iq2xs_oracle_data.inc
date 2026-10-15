// 本文件由 measure/iq2xs_oracle.py 生成，勿手改。
// 期望值来自 gguf-py 的 IQ2_XS.dequantize_blocks（独立实现路径，见 S-35）。
// 合成块规则：block[0:2] = fp16(1.5) 小端；block[k] = (k*37+11) & 0xff。
// 位型切片：out[0..15] + out[128..143] + out[240..255]，共 48 个。
const char kOracleBlockHex[] = "003e557a9fc4e90e33587da2c7ec11365b80a5caef14395e83a8cdf2173c6186abd0f51a3f6489aed3f81d42678cb1d6fb20456a8fb4d9fe23486d92b7dc01264b7095badf04294e7398";
const std::uint32_t kOracleBits[48] = {
    0xc1840000u, 0x424e4000u, 0xc1840000u, 0xc2b16000u, 0xc1840000u, 0xc24e4000u,
    0x41840000u, 0xc1840000u, 0x41840000u, 0xc24e4000u, 0x424e4000u, 0x41840000u,
    0x424e4000u, 0xc1840000u, 0xc24e4000u, 0xc1840000u, 0xc1e40000u, 0x43193000u,
    0xc1e40000u, 0xc3193000u, 0x41e40000u, 0x43193000u, 0x43193000u, 0xc1e40000u,
    0x42b22000u, 0xc2b22000u, 0x42b22000u, 0x42b22000u, 0xc3193000u, 0xc1e40000u,
    0x41e40000u, 0xc1e40000u, 0xc3193000u, 0xc1e40000u, 0x41e40000u, 0x41e40000u,
    0xc1e40000u, 0x41e40000u, 0x41e40000u, 0xc1e40000u, 0x41e40000u, 0x42b22000u,
    0x43193000u, 0xc1e40000u, 0xc1e40000u, 0xc2b22000u, 0x41e40000u, 0xc1e40000u,
};
const double kOracleL1 = 15164.625;
