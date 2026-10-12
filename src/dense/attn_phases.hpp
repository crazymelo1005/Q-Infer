// 注意力那一段的耗时切分（只作观测，给了才计时）。两类注意力的层各填自己有的那几项，互斥且穷尽：
//
//   GDN：proj 三个量化投影 / prep 前段（卷积 + SiLU + q/k 归一 + 两个门）/
//        rec 递推（beta 门 + gdn_step）/ close 收尾（rms·sigmoid）
//   QSA：proj 四个投影（含输出投影）/ prep q/门拆分与 q/k 归一 + rope /
//        kv 写入与本 cell 的页搬入 / idx 索引器全链（两投影、池化、逐 cell 打分、选择）/
//        attend softmax GQA 与乘门 + 按页聚集
//
// 为什么值得拆：整步相位图里注意力与 MoE 量级相当（各约四成），而这两段里只有 proj 走的是
// 按行并行的池子（experts/ffn 的 run_gemv），其余全是手写标量循环。要把注意力弄快，先得知道
// 钱花在哪一小段上——这正是 MoE 那次（MoePhases）用过的办法。
#pragma once

namespace qinfer::dense {

struct AttnPhases {
    double proj_ms = 0.0;
    double prep_ms = 0.0;
    double rec_ms = 0.0;
    double close_ms = 0.0;
    double kv_ms = 0.0;
    double idx_ms = 0.0;
    double attend_ms = 0.0;
};

}  // namespace qinfer::dense
