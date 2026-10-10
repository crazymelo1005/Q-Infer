// 本文件由 measure/iq4nl_oracle.py 从真实 GGUF 生成，勿手改。
// 张量 blk.0.ffn_down_exps.weight dims=[640, 2560, 512]，取前 16 个 IQ4_NL 块（每块 18 字节）。
// 期望值来自 gguf-py 的 IQ4_NL.dequantize_blocks（独立实现路径，见 S-40）。
const char kIq4nlRealHex[] =
    "d78b49970fc5e71dabf8c69d927e9264aa2d098de747c726a29a59ba5986aa80c7adb7a29d8974c20423206c5b8e3872"
    "eeff54e32432ae8bc78b30b77f98a08cbb34bbe14886277dd30a19366b50daa77167cddc65dbfb294c44c28a6e77d6cb"
    "91cb3865ea5031d34e67462c720b9b4a729a63376048a4e645366d4abcbd258ac52f316ac4869c23689bf39ad9a2bbc6"
    "230b20c67a9d8a57535b5d79f6391a4b4eeac70df658b4499972c5213995094aca5baac9bf0ae4d9f944f42518ec8867"
    "de63c9b7985a1b0c7c763b488468f773898b0958a4b7b2e9a08a95d2c0d6843e3451a5f3f37a695f5f97620d4c963a6b"
    "5d98677baef254547878bc87b409aaee13d0a8db6ce35a58329d7cfaaad0090c9171c2d820d996cbba177cae8cc5db0f";
const std::uint32_t kIq4nlRealFirst16[16] = {
    0xbb4bd600u, 0x3b1ccc00u, 0xbcdd79c0u, 0x3c093280u, 0x3b1ccc00u, 0xbc873cc0u, 0xbc14f500u, 0xb97ae000u, 0x3bac7a00u, 0xbc873cc0u, 0x3ca2ad40u, 0xbcae6fc0u, 0x3ca2ad40u, 0x3c401380u, 0xbbc3ff00u, 0xbc873cc0u,
};
const std::uint32_t kIq4nlRealBlock7_16[16] = {
    0xbc22d480u, 0x3c7eff80u, 0x3c47b280u, 0x3b872e00u, 0xbc22d480u, 0xb944a000u, 0xbb1fc200u, 0x3c7eff80u, 0x3b872e00u, 0xbb1fc200u, 0xbcad9540u, 0xbb1fc200u, 0xbc53fc80u, 0xbb999d00u, 0xbbe97e00u, 0xbc22d480u,
};
const double kIq4nlRealSum = 0.06283271312713623;
const double kIq4nlRealSumAbs = 5.419072508811951;
