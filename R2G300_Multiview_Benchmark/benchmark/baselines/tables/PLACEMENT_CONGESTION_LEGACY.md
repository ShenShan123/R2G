# Placement拥塞：旧批次已完成结果

**批次：`legacy_L3H2_100_seed42`，与v3 L4/H3批次不同。** 此处补充的是已完成的placement拥塞历史实验：六模型×B/C/D/F，共24项；E视图不适用。GNN层数3、预测MLP层数2、hidden64、attention heads4、100轮/耐心100、训练seed42/划分seed42。原始输入来自旧数据入口；同构geometry关闭、异构geometry开启。以下均为验证选定checkpoint的最终Test R²。

这些结果作为独立批次展示，不填入v3缺失项；不能把两个批次直接当作只改变阶段的严格对照。

| 模型 | B | C | D | E | F |
|---|---|---|---|---|---|
| GAT | -0.078178 | 0.032853 | **0.084896** | N/A | 0.023333 |
| GINE | **0.168423** | **0.138810** | 0.017119 | N/A | 0.086456 |
| ResGatedGCN | 0.139412 | -0.096486 | -0.258328 | N/A | **0.086476** |
| HAN | -0.038757 | 0.012366 | 0.082029 | N/A | -0.244044 |
| HGT | -0.047715 | -0.084120 | -0.094314 | N/A | -0.027236 |
| RGCN | -0.078580 | -0.053461 | -0.044925 | N/A | 0.037212 |

列内最高值加粗；负R²保留，数值未截断为0。MAE/RMSE均无量纲。

## 24项详细结果

| 模型 | 视图 | Test R² | MAE | RMSE | 参数量 | 最佳轮数 | 原始结果 |
|---|---|---:|---:|---:|---:|---:|---|
| GAT | B | -0.078178 | 0.083316 | 0.108900 | 143746 | 24 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gat__B__seed42.json) |
| GAT | C | 0.032853 | 0.079503 | 0.103140 | 147970 | 12 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gat__C__seed42.json) |
| GAT | D | 0.084896 | 0.078458 | 0.100327 | 172162 | 24 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gat__D__seed42.json) |
| GAT | F | 0.023333 | 0.079838 | 0.103647 | 244226 | 24 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gat__F__seed42.json) |
| GINE | B | 0.168423 | 0.074965 | 0.095639 | 80386 | 52 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gine__B__seed42.json) |
| GINE | C | 0.138810 | 0.075497 | 0.097326 | 84610 | 52 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gine__C__seed42.json) |
| GINE | D | 0.017119 | 0.080479 | 0.103976 | 108802 | 5 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gine__D__seed42.json) |
| GINE | F | 0.086456 | 0.077703 | 0.100241 | 180866 | 12 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__gine__F__seed42.json) |
| ResGatedGCN | B | 0.139412 | 0.076124 | 0.097292 | 129730 | 24 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__resgatedgcn__B__seed42.json) |
| ResGatedGCN | C | -0.096486 | 0.084676 | 0.109820 | 133954 | 1 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__resgatedgcn__C__seed42.json) |
| ResGatedGCN | D | -0.258328 | 0.090402 | 0.117646 | 158146 | 1 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__resgatedgcn__D__seed42.json) |
| ResGatedGCN | F | 0.086476 | 0.077175 | 0.100240 | 230210 | 15 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__resgatedgcn__F__seed42.json) |
| HAN | B | -0.038757 | 0.084174 | 0.106890 | 108610 | 2 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__han__B__seed42.json) |
| HAN | C | 0.012366 | 0.082424 | 0.104227 | 99330 | 4 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__han__C__seed42.json) |
| HAN | D | 0.082029 | 0.080177 | 0.100484 | 125442 | 9 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__han__D__seed42.json) |
| HAN | F | -0.244044 | 0.092033 | 0.116977 | 184002 | 77 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__han__F__seed42.json) |
| HGT | B | -0.047715 | 0.083981 | 0.107350 | 280342 | 88 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__hgt__B__seed42.json) |
| HGT | C | -0.084120 | 0.085587 | 0.109199 | 222075 | 23 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__hgt__C__seed42.json) |
| HGT | D | -0.094314 | 0.085019 | 0.109712 | 282819 | 21 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__hgt__D__seed42.json) |
| HGT | F | -0.027236 | 0.083003 | 0.106296 | 292392 | 28 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__hgt__F__seed42.json) |
| RGCN | B | -0.078580 | 0.085056 | 0.108920 | 104266 | 4 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__rgcn__B__seed42.json) |
| RGCN | C | -0.053461 | 0.084353 | 0.107644 | 108594 | 13 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__rgcn__C__seed42.json) |
| RGCN | D | -0.044925 | 0.084286 | 0.107207 | 132474 | 5 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__rgcn__D__seed42.json) |
| RGCN | F | 0.037212 | 0.081605 | 0.102907 | 204642 | 6 | [JSON](../raw/legacy_L3H2_100_seed42__placement__congestion__rgcn__F__seed42.json) |
