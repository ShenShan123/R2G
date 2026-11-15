# R2G-300：数据集与六模型Baseline

此目录统一提供300个设计实例的五视图物理设计图数据，以及线长/拥塞两任务、六模型、placement/route两阶段的baseline代码与实验结果。

| 内容 | 入口 |
|---|---|
| 数据集 | [dataset/](dataset/README.md)：6000个PyG图，30设计族×10版本×4阶段×5视图；完整数据由本仓库Release提供 |
| 六模型与介绍 | [benchmark/](benchmark/README.md)：GAT、GINE、ResGatedGCN、HAN、HGT、RGCN |
| Baseline主表 | [两任务×六模型×两阶段×五视图](benchmark/baselines/tables/BASELINE.md) |
| Excel结果 | [baseline_tables.xlsx](benchmark/baselines/baseline_tables.xlsx) |

主表108项已有结果、12项E视图拥塞N/A。placement拥塞为旧批次L3/H2，其余三组为v3 L4/H3，均在表内标注。数据的placement名称对应floorplan输入截止点，标签来自最终route/post-route；详情见模型README。

完整数据约155GiB，压缩分卷文件通过同一仓库的Release下载，不放入普通Git历史。代码目录、数据说明和下载工具统一位于本目录，下载恢复后无需改变原图内容即可训练。

```bash
# 从本目录执行；需Python与zstd、tar
python dataset/download_dataset.py --extract
python benchmark/scripts/verify_release.py --data-root dataset/R2G_multiview_v3_dataset
python benchmark/scripts/run_benchmark.py --data-root dataset/R2G_multiview_v3_dataset --stage route --task congestion --model gat --view C --device cuda:0 --execute
```

欢迎在固定设计族划分上优化模型并贡献可复现结果。输入阶段可见性、标签有效掩码和实体权重应保持一致。
