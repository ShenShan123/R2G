# 新版同构 GNN 工程

支持 GAT、GINE、ResGatedGCN。来源是原工程实际用于新版实验的 `native_main.py` 与 `native_homogeneous.py`，不是早期的旧入口。数据读取已收进本工程，不需要复制 `here_model`。

## 五个文件各做什么

| 文件 | 主要入口 | 职责 |
|---|---|---|
| `dataset.py` | `TASK_REGISTRY`、`load_native_graph`、`target_tensors` | 标签契约、原生图读取、有效掩码和实体权重 |
| 同上 | `schema_from_sample`、`convert_sample_to_homogeneous` | PyG 官方同构化，保留类型 ID、节点偏移和独立监督索引 |
| 同上 | `split_by_family`、`fit_transform_records` | 设计族划分、Train 标签变换 |
| `encoder.py` | `HomogeneousFeatureEncoder` | 各类型特征投影、类型嵌入、全局特征 |
| `model.py` | `HomogeneousGNN.encode/predict` | GNN 消息传播及节点/边 MLP 预测 |
| `train.py` | `run_one_graph`、`train_formal` | 分块 Head、反向传播、验证选模、Test |
| `main.py` | `parse_args/main` | 配置、构建对象、调用训练、保存结果 |

## 输入 → 同构图 → GNN

原生输入仍然是 `HeteroData`。在 `dataset.py` 用 `HeteroData.to_homogeneous` 将不同类型的节点放入统一索引空间，保存 `node_type/edge_type`；不同宽度特征用 NaN 补齐。监督的节点/边索引单独映射，不能直接用传播边序号代替目标序号。

`encoder.py` 根据真实类型宽度切片，不把补齐列作为真实物理特征。连续特征做 signed-log 变换，附有限值掩码；每种节点/边类型各自投影到统一宽度，并加入类型嵌入。**同构 GNN 不等于完全删除了类型特征**；这里是共享同构消息传播算子，仍保留输入类型信息。

`model.py` 调用 PyG `GATConv/GINEConv/ResGatedGraphConv`。编码后的边属性每层都会传给卷积；外围保留残差、GELU、Dropout、LayerNorm。节点 Head 输入一个节点向量；边 Head 输入两个端点的拼接、乘积和绝对差，共 `4 × hidden_dim`。

异构工程是关系/类型化传播，且边特征先汇入端点；本工程是同构拓扑上的共享算子，边特征进入每层卷积。这些差异被保留，不能把二者差异全部归因于“是否保留类型”。

## 训练逻辑

按设计族划分 Train/Val/Test；同一设计族的版本不能跨集合。只用 Train 的有效目标拟合变换，保留监督权重。每个设计全图编码一次，Head 按 `target-chunk-size` 分块；预测阶段先对节点嵌入的叶子副本累积梯度，再一次反传骨干，最后裁剪梯度并优化。

`train_formal` 用 Val loss 选择参数，结束后恢复最佳参数计算 Test。`max-targets-per-design=0` 不截断标签；非零会限制评估目标，正式全量对比应保持 0。`--activation-checkpointing` 可用计算换激活显存；`--amp bfloat16` 仅对 CUDA 生效，默认不启用。

## 数据与运行

数据格式见 [Dataset/README.md](Dataset/README.md)。安装匹配本机的 PyTorch 后执行 `python -m pip install -r requirements.txt`。本工程不需要邻居采样扩展。

从本目录运行：

```bash
CUDA_VISIBLE_DEVICES=2 python main.py \
  --stage route --view B --task wirelength --model gine \
  --hidden-dim 64 --num-layers 4 --num-head-layers 3 --heads 4 \
  --dropout 0.1 --lr 0.001 --weight-decay 0.00001 \
  --epochs 100 --patience 100 --test-every 0 \
  --target-chunk-size 65536 --max-targets-per-design 0 \
  --seed 42 --split-seed 42 --device cuda:0
```

这里只暴露物理 GPU2，`cuda:0` 是进程内编号。不指定设备时使用 CPU。更换任务或视图只需修改参数；支持的任务由 `dataset.py::TASK_REGISTRY` 决定。`--data-root` 可覆盖 `Dataset/<VIEW>`；显式 root 应仅包含所选视图。

默认输出：`Result/<run_name>/{train.log,results.json,best_checkpoint.pt}`。JSON 保存配置、设计族划分、目标变换和原始单位的 Test MAE/RMSE/R²。检查点只有最佳权重，不是完整的断点续训状态。同名目录拒绝覆盖。

`--smoke-test` 只检验单图训练连通性，不能当作设计外泛化实验；它限制目标行数，但本同构实现仍保留整张输入图，真实大图 smoke 不一定省内存。
