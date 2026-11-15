# 复现协议与证据边界

## 三个独立结果批次

1. `v3_L4H3_100_seed42`：84项记录。其中route的线长30项与拥塞24项来自`training_projects`的v3/100轮批次；placement线长30项来自`outputs/L4H3_3seed_20260904`的原始v3冻结入口。geometry均关闭，hidden64、消息层4、MLP层3、attention heads4、lr0.001、weight_decay1e-5、SmoothL1、100轮/耐心100、模型seed42、划分seed42。L4/H3中的H3表示MLP层数，不是注意力头数。异构placement线长启用训练设计顺序shuffle，原值保留在运行配置中；route不启用。placement拥塞24项未找到v3完成记录，配置可运行但没有历史测试数值。
2. `v3_route_L4H3_50_3seeds`：仅route，54个受支持组合×训练seed42/43/44，共162项；最大50轮、耐心50、划分42。均值与标准差按全部三个训练种子计算，标准差ddof=1。GAT/C拥塞R²为0.6911094689±0.0076554320；不是验证分数，也不是挑选最优种子。
3. `legacy_L3H2_100_seed42`：旧数据入口的两阶段108项结果。消息层3、MLP层2、100轮、seed42；异构开启geometry，同构关闭。保留原始JSON和表格作为历史参考，不能冒充v3同协议基准，也未声称只凭本包v3复现其数字。旧输入与v3没有全张量等价证明。

## 可迁移代码与历史源

`models/homo_model`和`models/here_model`共十个核心Python文件与v3 Route的50/100轮计划中记录的源码哈希完全一致；未复制旧CVPR merged-data工程。原来额外保留的历史placement源码副本已移除；models只含homo_model与here_model两个工程，共六个模型。运行器统一调用这两个工程，历史placement成绩是独立来源的参考值，不宣称本包源码与其旧冻结实现完全相同。历史路径出现在原始结果JSON中仅为来源信息，不作为运行默认值。

`configs/benchmark_v3_L4H3_100_seed42.json`逐项保存原配置和关联原结果。默认只选择84项有历史记录的配置；显式`--include-unreported`才加入24项placement拥塞。E视图拥塞不加入队列。旧训练checkpoint只含最佳模型状态，不能宣称具有完整optimizer/RNG的断点续训能力。目标变换参数和划分保存在原始结果JSON中。本包不附权重。

## 评价与数据边界

按设计族分组而非随机Gate/Net行划分；240/30/30实例对应24/3/3设计族。验证族为i2c_verilog、pci、spi_top；测试族为des3_area、tinyRocket、usbf_top。所有任务均在最佳验证损失checkpoint上取最终Test，不取history中周期性Test的最大值。WireLength物理空间MAE/RMSE单位为µm；拥塞为无量纲。R²允许为负，缺失值不能填成0。不同视图的监督载体与有效实体集合不应未经对齐就认定完全相同。

本包不含数据或权重。请使用外部原有多视图数据目录；`verify_release.py --data-root <目录>`可核对manifest和6000个图文件路径是否齐全，不对未执行的张量或训练精度校验作保证。

## 本次验证范围

实际执行的检查和环境见`provenance/VALIDATION.json`。CPU合成检查验证模型和任务组合能执行前向、反向与一次参数更新；真实小图短测试仅验证数据可读及入口可运行。它们都不能替代完整300设计训练或证明新精度。跨PyTorch/CUDA/硬件环境不保证逐位复现。主结果来自已保存的历史实验，没有在此包装过程中重新跑100轮。

## 阶段名称与真实输入截止点

300/300个v3 placement B图metadata标记feature_cutoff=floorplan、coordinate_stage=floorplan、standard_cell_coordinates=false、gate_gate_geometry=false。共享标签来源为route_or_post_route。缺少标准单元输入坐标限制的是空间网格/几何关系构建，不禁止基于逻辑与单元特征的Gate拥塞回归。旧L3/H2已有24项placement拥塞完成结果；v3 L4/H3未找到完成记录，未核实原调度取消原因。详见PLACEMENT_CONGESTION.md。

## 最终Baseline展示规则

主表为两任务×六模型×两阶段×五视图，直接展示108项已有结果和12项E视图拥塞N/A。placement拥塞使用legacy_L3H2_100_seed42的24项，其余使用v3_L4H3_100_seed42的84项；每组标注批次。该合并展示不改变原始结果与同批次覆盖记录，不声称已完成v3 placement拥塞新训练。运行器的v3配置范围保持原样，不能拿混合批次主表当作统一配置的精度复现目标。
