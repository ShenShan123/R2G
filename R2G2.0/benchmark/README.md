# R2G-300：多视图GNN线长与拥塞预测基准

本项目整理300个设计实例（30个设计族×10个版本）的历史GNN训练实验，提供六模型代码、线长与拥塞baseline表格，以及固定数据划分和训练配置。模型包括GAT、GINE、ResGatedGCN、HAN、HGT、RGCN；比较placement和route两个输入阶段、B/C/D/E/F五种图视图，以预测最终布线后的物理指标，为电路物理设计性能预测与后续优化提供可扩展基线。欢迎大家使用我们的数据集，改进特征编码、消息传播与任务读出，提交可复现的模型和实验结果。

本目录仅包含**代码、表格、文档和结果证据**，不复制数据集或模型权重。原工程与实验来源为`/data1/zhouzw/CVPR_modelR2G2.0`；训练时通过`--data-root`指定外部多视图数据目录。本机已存在的统一v3数据入口为`/data/R2G_multiview_v3_dataset`，需包含`manifest.csv`及`samples/`，原工程根目录本身不是该数据入口。

## Baseline表格

**统一呈现：两个任务（线长、拥塞）×六模型×两个阶段（placement、route）×五视图（B/C/D/E/F）。**

| 文件 | 内容 |
|---|---|
| [Baseline主表](baselines/tables/BASELINE.md) | 四组6×5矩阵：placement线长、placement拥塞、route线长、route拥塞，列内最好值加粗 |
| [Baseline Excel](baselines/baseline_tables.xlsx) | 前四个工作表就是上述主表；后续保留历史与三种子明细 |
| [Baseline主表CSV](baselines/baseline_120_slots.csv) | 120个位置，含批次、R²/MAE/RMSE、参数量、配置及原始结果链接 |
| [v3 Route三种子表](baselines/tables/V3_ROUTE_3SEED_50EPOCH.md) | 独立补充：50轮、训练seed42/43/44，固定split42，162项结果的均值±样本标准差 |
| [全部历史明细](baselines/all_results.csv) | 保留354项已归档结果，供追溯 |

主表共**108项已有结果＋12项N/A**。E视图没有Gate，两个阶段的原Gate拥塞任务均不适用。**Placement拥塞直接采用旧批次`legacy_L3H2_100_seed42`的24项完成结果，已填入主表和Excel对应位置；其余三组采用`v3_L4H3_100_seed42`。** 每组标题标注批次。合并展示不代表两批输入数据和模型设置完全一致，也不表示补跑了v3 placement拥塞。

v3严格同批次尚缺24项placement拥塞记录，原覆盖表单独保留在`baselines/v3_120_slots.csv`。此前约0.69的拥塞成绩来自50轮三种子Route实验：GAT/C Test R²为0.6911±0.0077；该补充批次未混入100轮主表。

固定`split_seed=42`按设计族隔离：Train/Val/Test为240/30/30实例，同族十个版本不跨集合。WireLength是每个物理Net的最终布线长度（µm），不是整图总线长；Congestion是Gate级最终拥塞代理值（无量纲）。表格取验证损失选定checkpoint的最终Test指标。有效掩码、实体权重及只由Train拟合的目标变换由原代码保留。主表placement线长的历史成绩来自早期冻结源码，异构侧启用训练设计顺序shuffle；本包统一只保留六模型整理版代码及原配置，不再附该历史源码副本。复跑这些配置不承诺与历史实现逐位一致。

## 使用

推荐Python 3.9；本次CPU检查使用PyTorch 2.5.1、PyG 2.6.1、NumPy 2.0.2。根据服务器选择匹配CUDA构建。历史环境与复现边界见[说明](docs/REPRODUCIBILITY.md)。

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py

# 默认打印命令，不占用GPU；数据目录可迁移到其他服务器
python scripts/run_benchmark.py --data-root /data/R2G_multiview_v3_dataset --stage route --task congestion --model gat --view C
# 正式运行：恢复原始配置，不覆盖既有同名结果
python scripts/run_benchmark.py --data-root /data/R2G_multiview_v3_dataset --stage route --task congestion --model gat --view C --device cuda:0 --execute
```

运行器支持六模型、两个阶段及各任务的有效视图；默认只选择84项有记录的v3主表配置。`--include-unreported`可加入24项尚无历史成绩的placement拥塞配置开展新实验。E视图拥塞不进入队列。输出为`runs/`，不具备完整optimizer/RNG断点续训功能。更早的L3/H2旧入口表是独立历史参考，未证明其旧数据与v3逐张量等价，不承诺只用v3精确复现该表。

## 模型优化方向

欢迎研究：①不同节点/边类型的物理特征编码与边感知注意力；②关系门控、深层残差传播和面向Net/Gate的任务读出；③明确实体对齐后的多视图融合；④在固定设计划分下改进泛化、尾部误差与多种子稳定性。训练时只能使用输入阶段可见信息，不能将最终标签作为特征；超参数由验证集选择，测试集用于最终评估。请同时报告数据版本、阶段、视图、训练/划分种子、模型参数量和R²/MAE/RMSE，便于公平比较。

目录：`models/`仅有`homo_model/`（GAT/GINE/ResGatedGCN）和`here_model/`（HAN/HGT/RGCN）两个工程，`configs/`为划分和训练配置，`baselines/`为表格及354份原始结果，`scripts/`为运行和校验工具，`tests/`为CPU小图检查，`provenance/`为来源哈希和验证记录。本次整理未重新训练完整基准，也未上传GitHub。

## Placement拥塞与坐标的核实结论

- **不能说placement从未跑过拥塞**：旧L3/H2历史批次有24项已完成的placement拥塞实验，已列入独立参考表。v3 L4/H3批次未找到相应完成记录，因此主表没有这些测试数字，也没有证据仅凭此认定当时未运行的原因。
- **300个v3样本的placement输入截止点实际上是floorplan**：逐一检查B视图metadata，300/300均标记`standard_cell_coordinates=false`、`gate_gate_geometry=false`；标准单元尚无可信布局坐标，无法在该输入时点据其位置建立网格几何关系。阶段名称不能代替实际输入截止点。
- **坐标缺失不等于最终拥塞标签缺失**：标签由route/post-route结果按Gate实体对齐到早期图，模型无需在floorplan位置重新计算真值。可利用逻辑拓扑、单元及全局特征预测最终Gate标签，但不能把后端坐标当作早期输入，也不能把该输出冒称为已知位置的拥塞热图。
- **E视图拥塞不适用**：E没有Gate，原任务没有监督载体，与标准单元坐标是否存在是两个不同问题。

核对证据见[placement拥塞说明](docs/PLACEMENT_CONGESTION.md)及`provenance/placement_congestion_audit.json`。
