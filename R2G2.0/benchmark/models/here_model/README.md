# 新版异构 GNN 工程

支持 HGT、RGCN、HAN。五个核心文件完全自包含，不依赖同级其他工程或原来的 `here_model`。

## 阅读代码的顺序

1. `main.py::parse_args/main`：选择阶段、视图、任务、模型与设备。
2. `dataset.py::TASK_REGISTRY`：任务单位、标签列名、目标变换。
3. `dataset.py::discover_records/load_native_graph/target_tensors`：数据发现、传播关系、有效监督与权重。
4. `dataset.py::split_records_by_family/fit_transform_from_records`：按设计族隔离数据；只在 Train 拟合标签变换。
5. `encoder.py::NativeViewFeatureEncoder`：不同类型独立投影；边特征均值汇入端点；加入类型及全局特征。
6. `model.py::build_model/NativeTaskHead`：选择 PyG 骨干与节点/边预测头。
7. `train.py::train_multidesign/run_design_epoch`：逐设计优化、分块 Head、验证选模、最终 Test。

## 数据和张量如何变化

输入 `HeteroData` 包含不同节点类型的 `x`、不同关系的 `edge_index/edge_attr`、标签和掩码。节点特征宽度不必相同。编码后，各类型成为 `[N_type, hidden_dim]`。

连续特征默认 `sign(x) * log1p(abs(x))`，并附有限值掩码；缺失数值不能仅替换为 0 后不加区分。编码器对各关系的边特征独立投影，并汇入源/目标节点，因此“没有 GNN 传播”也不等于“完全没有结构信息”。

HGT/HAN 使用类型化节点和关系；RGCN 将节点展平后保留关系 ID。三者都有原有的残差、GELU、Dropout 和 LayerNorm。预测节点目标时读取该节点向量；预测边目标时读取 `[h_src, h_dst, h_src*h_dst, |h_src-h_dst|]`，交给 MLP 回归。Cc 的可选对称 Head/几何特征路径也保留。

`target_tensors` 将有效掩码与有限值检查组合。复制到多条边上的物理实体标签按 `entity_weight` 加权，不能把多条副本当成多份独立实体。监督边与传播边分别定义，时序/Cc 等查询边不会仅因含标签就自动参与消息传递。

## 训练逻辑

正式训练至少需要三个设计族。先划分 Train/Val/Test，再排除没有有效目标的设计；不会为了某任务重新移动设计族。标签变换只用训练集合拟合。每个设计做一次全图前向；Head 可分块计算，累积梯度后回传 GNN，控制预测头显存，**不是对标签抽样**。

默认 AdamW、SmoothL1、梯度裁剪。验证 loss 或显式设置的验证 R² 用于选择检查点；Test 不参与选择。`--test-every 0` 关闭训练中的 Test 监控，只在最后评估所选模型。结果用原始标签单位报告 MAE、RMSE、R²。`--smoke-test` 是单设计内部行划分的连通性测试，不是论文实验划分。

## 输入与运行

数据布局见 [Dataset/README.md](Dataset/README.md)。依赖安装：先安装与本机匹配的 PyTorch，再执行 `python -m pip install -r requirements.txt`。

从本目录运行，以下是显式 L4/H3 的命令示例，不会自动启动批量实验：

```bash
# 只暴露物理 GPU1；进程里的 cuda:0 就是物理 GPU1，不会用物理 GPU0。
CUDA_VISIBLE_DEVICES=1 python main.py \
  --stage route --view B --task wirelength --model rgcn \
  --hidden-dim 64 --num-layers 4 --num-head-layers 3 --heads 4 \
  --dropout 0.1 --lr 0.001 --weight-decay 0.00001 \
  --epochs 100 --patience 100 --test-every 0 \
  --target-chunk-size 65536 --max-targets-per-design 0 \
  --seed 42 --split-seed 42 --device cuda:0
```

换物理 GPU2 时仅将环境变量改为 `CUDA_VISIBLE_DEVICES=2`。不写 `--device` 默认 CPU。`--data-root /path/to/B_catalog` 可覆盖默认路径；同一 root 应只包含所选视图，不要将含 B–F 五套图的总目录直接作为单视图 root。

可选先做输入审计：`python main.py --view B --stage route --inspect-only`。查看全部参数：`python main.py --help`。

## 输出与注意事项

`Result/<run_name>/train.log`、`results.json`、`best_checkpoint.pt`。配置、数据划分、标签变换和指标记录在 JSON；默认运行名包含时间戳。显式运行名重复会拒绝覆盖。检查点不是完整断点恢复包，未保存全部优化器和 RNG 状态。

任务 `cg` 是对地电容，不是拥塞；拥塞使用 `congestion`，当前 E 视图没有该任务。`setup_slack/hold_slack` 虽在五个视图可读，但 F 主要保留 IO 端点，与其他视图全部 Pin/IO 标签不天然公平对齐。
