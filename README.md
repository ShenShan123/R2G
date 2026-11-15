# R2G2.0: Multi-view Circuit Graph Dataset and GNN Baselines

R2G2.0 provides 300 circuit design instances (30 design families × 10 variants), five native PyG graph views, and six GNN baselines for routed wirelength and congestion prediction from placement/route inputs.

**New-version entry: [R2G2.0/](R2G2.0/README.md)**

| Resource | Link |
|---|---|
| Dataset and download tool | [R2G2.0/dataset](R2G2.0/dataset/README.md) |
| Six-model code and usage | [R2G2.0/benchmark](R2G2.0/benchmark/README.md) |
| Two tasks × six models × two stages × five views | [Baseline results](R2G2.0/benchmark/baselines/tables/BASELINE.md) |
| Excel results | [baseline_tables.xlsx](R2G2.0/benchmark/baselines/baseline_tables.xlsx) |

Models: GAT, GINE, ResGatedGCN, HAN, HGT and RGCN. The main table contains 108 recorded experiment results and 12 unsupported E-view congestion entries. Placement congestion uses an explicitly labelled earlier L3/H2 batch; the other three matrices use v3 L4/H3. These are historical results, not newly rerun unified-configuration measurements. Input cutoff and target provenance are documented in the benchmark README.

The full dataset contains floorplan/placement/cts/route × B/C/D/E/F = 6000 graphs. Large dataset assets are distributed through GitHub Releases and verified by SHA256. **The complete dataset release is currently being uploaded; code and baseline tables are available now.**

## Original version / 旧版本

The complete repository before this update is preserved as **R2G1.0**, at original commit `a2c6ea1f110946f7f1ca30bb117ad83061fce981`:

- [Browse all original files](https://github.com/ShenShan123/R2G/tree/R2G1.0)
- [Original-version release](https://github.com/ShenShan123/R2G/releases/tag/R2G1.0)
- [Download original source ZIP](https://github.com/ShenShan123/R2G/archive/refs/tags/R2G1.0.zip)
- [Download original source tar.gz](https://github.com/ShenShan123/R2G/archive/refs/tags/R2G1.0.tar.gz)

旧版完整文件保存在R2G1.0标签和Release中，含原始数据管线、gnn-node/gnn-edge、数据索引、提取脚本及说明。默认分支main展示R2G2.0；不需额外网盘即可访问和下载旧版快照。
