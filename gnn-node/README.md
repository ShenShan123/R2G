# R2G: Multi-View Circuit Graph Benchmark Suite

CVPR 2026 Reproduction Code

## 📋 Project Overview

R2G is a standardized circuit graph benchmark suite that converts DEF files into typed, information-preserving circuit graphs, supporting placement and routing tasks.

### Key Features

- **Multi-view Circuit Graphs**: Five complementary views (b-f) with information equivalence
- **Stage-aware Tasks**: Placement (node-level) and routing (edge-level) with matching resolution supervision
- **Standardized Evaluation**: Unified splitting, metrics, and reproducible baselines
- **End-to-end Pipeline**: DEF to graph conversion, loaders, metrics, and baselines

### Paper Key Findings

1. **View Selection is Critical**: View (b) performs best on both placement and routing
2. **GNN Depth**: 3-4 layers suffice; deeper leads to over-smoothing
3. **Head Depth**: 3-4 layers are key for accuracy and stability
4. **Model-View Interaction**: Different models rank differently on different views

## 📁 Project Structure

```
R2G/
├── main.py                  # Entry point: argument parsing and training pipeline
├── model.py                 # GNN model architecture (encoder + GNN + decoder head)
├── dataset.py               # Dataset class: loading, preprocessing, normalization
├── encoders.py              # Feature encoders for nodes and edges
├── sampling.py              # Data sampling and train/val/test split creation
├── downstream_train.py      # Training loop, evaluation, and checkpointing
└── data/                    # Data directory
    ├── raw/                 # Raw .pt files
    ├── processed_node_regression/
    ├── processed_edge_regression/
    └── ...
```

---

## File-by-File Documentation

### 1. `main.py` - Entry Point

**Purpose**: Orchestrate entire training pipeline from argument parsing to model training.

**Key Functions**:
- `parse_args()`: Parse all command-line arguments for model configuration
- `log_and_save_run_info()`: Log training metadata (split info, hyperparameters)
- `main()`: Main training pipeline

**Data Flow**:
```
Command-line arguments
       ↓
parse_args() → args object
       ↓
Setup logging
       ↓
Set random seeds
       ↓
Load dataset (MergedHomographDataset)
       ↓
Sample data (dataset_sampling)
       ↓
Create model (GraphHead)
       ↓
Train model (downstream_train)
       ↓
Save results and plots
```

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | `data/` | Dataset root directory |
| `--dataset` | `place_B_homograph` | Dataset name |
| `--task_level` | `node` | Task level: node/edge/graph |
| `--task` | `regression` | Task type: regression/classification |
| `--model` | `resgatedgcn` | GNN type: resgatedgcn/gat/gine |
| `--num_gnn_layers` | `3` | GNN message passing layers (3-4 recommended) |
| `--num_head_layers` | `2` | Head MLP layers (3-4 recommended for best performance) |
| `--hid_dim` | `128` | Hidden layer dimension |
| `--lr` | `0.001` | Learning rate |
| `--epochs` | `100` | Training epochs |
| `--gpu` | `0` | GPU index |

**Usage Example**:
```bash
python main.py \
    --dataset route_B_homograph \
    --task_level edge \
    --task regression \
    --model gine \
    --num_gnn_layers 4 \
    --num_head_layers 3 \
    --hid_dim 256 \
    --lr 0.0001 \
    --epochs 100 \
    --gpu 0
```

---

### 2. `model.py` - GNN Architecture

**Purpose**: Implement complete GNN model for circuit graph prediction tasks.

**Key Class**: `GraphHead`

**Architecture Components**:
```
Input Batch (raw features)
       ↓
FeatureEncoder (node + edge encoders)
       ↓
Encoded embeddings
       ↓
GNN Layers (message passing, K layers)
       ↓
Node embeddings
       ↓
Task-specific processing
    ├─ Node tasks: head_layers(node_emb)
    ├─ Edge tasks: aggregate(src, dst, edge) → head_layers()
    └─ Graph tasks: pool → head_layers()
       ↓
Predictions
```

**Forward Pass Details**:

**For Node-Level Tasks**:
1. Encode features → `[num_nodes, hidden_dim]`
2. Apply GNN layers → `[num_nodes, hidden_dim]`
3. Apply head MLP → `[num_nodes, output_dim]`
4. Filter valid nodes (labels != -1)

**For Edge-Level Tasks**:
1. Encode features → `[num_nodes, hidden_dim]`, `[num_edges, hidden_dim]`
2. Apply GNN layers → `[num_nodes, hidden_dim]`
3. Gather embeddings for labeled edges:
   - `src_emb = x[edge_index[0]]`
   - `dst_emb = x[edge_index[1]]`
   - `edge_attr_emb` (from edge embeddings)
4. Aggregate (concat/add/mean) → `[num_labeled_edges, 2*hidden_dim]`
5. Apply head MLP → predictions

**Supported GNN Types**:
- `resgatedgcn`: Residual Gated Graph Convolution with edge features (most stable)
- `gat`: Graph Attention Network with edge features (specialized for view e)
- `gine`: Graph Isomorphism Network with MLP and edge features (best overall)

**Node Types** (4 types):
- `gate` (0): Logic gate unit
- `io_pin` (1): I/O pin (top-level port)
- `net` (2): Network connecting multiple nodes
- `pin` (3): Input/output pin of logic gate

**Edge Types** (8 types):
- `gate_gate`, `gate_net`, `gate_pin`
- `io_pin_gate`, `io_pin_net`, `io_pin_pin`
- `pin_net`, `pin_pin`

---

### 3. `dataset.py` - Data Loading and Preprocessing

**Purpose**: Load, preprocess, and normalize circuit graph data.

**Key Class**: `MergedHomographDataset`

**Data Processing Pipeline**:
```
Raw .pt file (x, y, edge_index, edge_attr, global_features, global_y)
       ↓
Load components
       ↓
Determine train/test masks (based on graph_ids)
       ↓
Apply normalization (regression only):
   1. Compute offset (if min <= 0)
   2. Log transform: log(x + offset + epsilon)
   3. Standardize: (x - mean) / std
   4. Save params for inverse transform
       ↓
Concatenate global features:
   - Node: [9 dims] + [5 dims] → 14 dims
   - Edge: [9 dims] + [5 dims] → 14 dims
       ↓
Create train/val/test masks
       ↓
Convert to undirected graph
       ↓
Save processed data (cached for fast reload)
```

**Key Features**:

1. **Normalization Strategy**:
   - Only uses training data to compute mean/std (prevents data leakage)
   - Handles zero/negative values with offset
   - Supports both forward and inverse transforms

2. **Data Splits**:
   - Test set: Fixed IDs from `--fixed_test_ids` or random split
   - Validation set: Fixed IDs from `--fixed_val_ids` or random split
   - Training set: Remaining designs

3. **Valid Label Handling**:
   - Uses sentinel value -1 for invalid/no-label samples
   - Only samples with labels != -1 are used for training/evaluation
   - `valid_mask` tracks which samples are valid

4. **Caching**:
   - Processed data is saved to `processed_{task_level}_{task_type}/` directory
   - Subsequent runs load cached data (much faster)

**Standardization Process**:
```python
# Forward transform
offset = -min(0, global_min) + epsilon
y_log = log(y + offset)
y_norm = (y_log - mean) / std

# Inverse transform (for evaluation)
y_log = y_norm * std + mean
y_orig = exp(y_log) - epsilon - offset
```

---

### 4. `encoders.py` - Feature Encoding

**Purpose**: Transform raw circuit features into learned embeddings.

**Key Classes**:

1. **`HomographNodeEncoder`**:
   - Processes node features separately for each node type
   - Handles discrete (categorical) and continuous features differently
   - Discrete: Embedding lookup (learned vectors)
   - Continuous: Linear projection

2. **`HomographEdgeEncoder`**:
   - Similar to node encoder but for edge features
   - Processes each edge type separately

3. **`FeatureEncoder`**:
   - Combines node and edge encoders
   - Applies optional batch normalization

**Encoding Process for Nodes**:
```
Raw node features [num_nodes, feature_dim]
       ↓
Group by node_type
       ↓
For each node type:
   ├─ Discrete features: Embedding lookup → [num_nodes_type, emb_dim]
   └─ Continuous features: Linear projection → [num_nodes_type, emb_dim]
       ↓
Sum discrete + continuous
       ↓
Combine all node types
       ↓
Encoded embeddings [num_nodes, hidden_dim]
```

**Discrete vs Continuous Features**:

**Discrete (Categorical)**:
- Examples: cell_type (96 categories), orientation (8 categories), net_type (6 categories)
- Encoded via: `nn.Embedding(num_categories, embedding_dim)`
- Learned representations for each category

**Continuous (Numerical)**:
- Examples: x, y coordinates, area, power leakage
- Encoded via: `nn.Linear(input_dim, hidden_dim)`
- Linear transformation + learned weight/bias

---

### 5. `sampling.py` - Data Sampling

**Purpose**: Create train/val/test data loaders with neighbor sampling for GNN training.

**Key Functions**:

1. **`dataset_sampling()`**: Main function to create data loaders
2. **`get_nodes_from_subgraphs()`**: Sample nodes from specified graphs
3. **`plot_true_values_distribution_before_sampling()`**: Visualize label distribution

**Data Flow**:
```
MergedHomographDataset (full graph with all designs)
       ↓
Split graphs into train/val/test sets
       ↓
Sample nodes from each split (label != -1)
       ↓
Create NeighborLoader data loaders
       ↓
Return loaders + max_label
```

**Neighbor Sampling**:

**Why Sample?**
- Large circuit graphs have millions of nodes/edges
- Full-batch training is infeasible
- Neighbor sampling creates manageable subgraphs for each batch

**How It Works**:
```
Target nodes (e.g., 1024 nodes for batch)
       ↓
Sample neighbors within K hops:
   ├─ Hop 1: Sample N neighbors for each target node
   ├─ Hop 2: Sample N neighbors for each hop-1 node
   └─ ...
       ↓
Create subgraph containing target nodes + sampled neighbors
       ↓
Process only this subgraph in GNN
```

**Sampling Parameters**:
- `num_hops` (default 2): Number of neighbor hops to sample
- `num_neighbors` (default 10): Number of neighbors to sample per hop
- `sample_ratio` (default 1.0): Fraction of nodes to use
- `batch_size` (default 1024): Number of targets per batch

---

### 6. `downstream_train.py` - Training and Evaluation

**Purpose**: Implement training loop, evaluation metrics, and model checkpointing.

**Key Functions**:

1. **`train_epoch()`**: Train for one epoch
2. **`evaluate()`**: Evaluate model on a dataset
3. **`downstream_train()`**: Main training function with early stopping

**Training Loop**:
```
Initialize model, optimizer, scheduler, criterion
       ↓
For each epoch:
   ├─ Train for one epoch (train_epoch)
   │   └─ For each batch:
   │       ├─ Forward pass
   │       ├─ Compute loss (only on valid labels)
   │       ├─ Backward pass
   │       └─ Update weights
   ├─ Evaluate on train set
   ├─ Evaluate on validation set (evaluate)
   │   └─ Compute metrics (MAE, RMSE, R² for regression)
   ├─ Update learning rate (scheduler.step())
   └─ If validation improves:
       ├─ Evaluate on train and test sets
       ├─ Save model checkpoint
       └─ Record best metrics
       ↓
Load best model and final evaluation
       ↓
Return best test metrics
```

**Training Details**:

**Loss Functions**:
- Regression: `SmoothL1Loss` (Huber loss - robust to outliers)
- Classification: `CrossEntropyLoss`

**Optimization**:
- Optimizer: Adam
- Learning Rate: 0.001 (default)
- Weight Decay: 1e-5 (L2 regularization)

**Learning Rate Schedulers**:
- `plateau`: ReduceLROnPlateau (adaptive, reduces when validation loss stagnates)
- `cosine`: CosineAnnealingLR (smooth decay to minimum)
- `step`: StepLR (step-based decay)

**Evaluation Metrics**:

**Regression**:
- MAE (Mean Absolute Error): Average absolute difference
- RMSE (Root Mean Squared Error): Square root of average squared error
- R² (Coefficient of Determination): Proportion of variance explained

**Classification**:
- Accuracy: Proportion of correct predictions
- F1 Score: Weighted average of precision and recall
- Classification Report: Per-class precision, recall, F1

---

## Data Flow Diagram

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                           │
│  1. Parse arguments (--dataset, --model, --num_gnn_layers...)  │
│  2. Setup logging (train.log)                                   │
│  3. Set random seeds (torch, numpy)                             │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    dataset.py                              │
│  1. Load raw data from data/raw/<dataset>.pt                  │
│  2. Split train/val/test (by graph_id)                        │
│  3. Normalize labels (log + standardize, train only)           │
│  4. Concatenate global features (node + global)                │
│  5. Create valid mask (labels != -1)                           │
│  6. Convert to undirected graph                              │
│  7. Save to processed_<task_level>_<task_type>/                │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    sampling.py                             │
│  1. Sample nodes from train/val/test graphs                   │
│  2. Create NeighborLoader (neighbor sampling)                  │
│  3. Plot label distribution                                    │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                     model.py                               │
│  ┌─────────────────────────────────────────────┐            │
│  │              FeatureEncoder                │            │
│  │  ├─ HomographNodeEncoder (encoders.py)    │            │
│  │  │   ├─ Discrete: Embedding               │            │
│  │  │   └─ Continuous: Linear               │            │
│  │  └─ HomographEdgeEncoder (encoders.py)    │            │
│  │      ├─ Discrete: Embedding               │            │
│  │      └─ Continuous: Linear               │            │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│  │            GNN Layers (K layers)        │            │
│  │  (message passing: GINE/GAT/ResGatedGCN)   │         │
│  │  Supports edge features                      │         │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│  │  Task-Specific Processing                   │            │
│  │  ├─ Node tasks: head_layers(node_emb)       │            │
│  │  └─ Edge tasks: agg(src,dst,edge)→head()   │            │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│              Predictions                              │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                downstream_train.py                          │
│  1. Setup optimizer, scheduler, criterion                  │
│  2. Training loop:                                        │
│     ├─ train_epoch: Forward → Loss → Backward → Update    │
│     ├─ evaluate on train set                               │
│     ├─ evaluate on val set                                │
│     └─ Save best model (by val loss)                    │
│  3. Load best model and final evaluation                   │
│  4. Save results to .npz files                            │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
              Results and Logs
```

---

## Module Interactions

### Function Call Hierarchy

```
main()
├── parse_args()
├── MergedHomographDataset.__init__() [dataset.py]
│   └── MergedHomographDataset.process() [dataset.py]
│       ├── log_transform() [dataset.py]
│       ├── standardize() [dataset.py]
│       └── (save to cache)
├── dataset_sampling() [sampling.py]
│   ├── print_distribution_stats() [sampling.py]
│   ├── get_nodes_from_subgraphs() [sampling.py]
│   └── NeighborLoader() [PyG]
├── GraphHead.__init__() [model.py]
│   └── FeatureEncoder.__init__() [encoders.py]
│       ├── HomographNodeEncoder.__init__() [encoders.py]
│       └── HomographEdgeEncoder.__init__() [encoders.py]
├── downstream_train() [downstream_train.py]
│   ├── train_epoch() [downstream_train.py]
│   │   └── GraphHead.forward() [model.py]
│   │       └── FeatureEncoder.forward() [encoders.py]
│   │           ├── HomographNodeEncoder.forward() [encoders.py]
│   │           └── HomographEdgeEncoder.forward() [encoders.py]
│   └── evaluate() [downstream_train.py]
│       ├── GraphHead.forward() [model.py]
│       └── inverse_standardize(), inverse_log_transform() [dataset.py]
└── (save scatter plots)
```

### Data Transformations

1. **Raw Data → Processed Data** [dataset.py]:
   ```
   x: [N, 10] → [N, 14] (concatenate global features)
   y: [N,] → [N,] (normalized for regression)
   edge_index: [2, E] → [2, E] (converted to undirected)
   edge_attr: [E, 10] → [E, 14] (concatenate global features)
   ```

2. **Raw Features → Embeddings** [encoders.py]:
   ```
   Node features: [N, 14] → [N, hidden_dim]
   Edge features: [E, 14] → [E, hidden_dim]
   ```

3. **Embeddings → Predictions** [model.py]:
   ```
   Node tasks: [N, hidden_dim] → [N, 1] (regression)
   Edge tasks: [E, 2*hidden] → [E, 1] (regression)
   ```

4. **Normalized → Original Scale** [dataset.py, downstream_train.py]:
   ```
   y_norm → y_original:
     1. y_std = y_norm * std + mean
     2. y_log = exp(y_std) - epsilon
     3. y_orig = y_log - offset
   ```

---

## How to Reproduce Experiments

### Quick Start

1. **Prepare Data**:
   - Place processed `.pt` files in `data/raw/`
   - Dataset naming: `place_B_homograph.pt`, `route_B_homograph.pt`, etc.

2. **Run a Single Experiment**:
   ```bash
   # Node-level task (Placement)
   python main.py \
       --dataset place_B_homograph \
       --task_level node \
       --model resgatedgcn \
       --num_gnn_layers 3 \
       --num_head_layers 3 \
       --hid_dim 256 \
       --lr 0.0001 \
       --epochs 100 \
       --gpu 0

   # Edge-level task (Routing)
   python main.py \
       --dataset route_B_homograph \
       --task_level edge \
       --model gine \
       --num_gnn_layers 4 \
       --num_head_layers 3 \
       --hid_dim 256 \
       --lr 0.0001 \
       --epochs 100 \
       --gpu 0
   ```

### Understanding Output

**Log Files** (`results/{dataset}_{model}_{task_level}/train.log`):
- Contains complete training progress
- Includes: configuration, epoch-by-epoch metrics, final test results

**Model Checkpoints** (`results/{dataset}_{model}_{task_level}/best_model.pt`):
- Best model weights (by validation loss)

**Test Results** (`results/{dataset}_{model}_{task_level}/test_results.npz`):
- NumPy arrays with predictions and labels

**Scatter Plots** (`results/{dataset}_{model}_{task_level}/plots_*/`):
- Train/Val/Test scatter plots (true vs predicted values)

---

## 📊 Experiment Results

### Key Findings from CVPR 2026 Paper

#### 1. View Selection

**View (b)** performs best on both placement and routing:
- Placement: GINE R² ≈ 0.89, ResGatedGCN R² ≈ 0.88
- Routing: GINE R² ≈ 0.87, ResGatedGCN R² ≈ 0.88

**View Priority**:
- Placement: (b) > (c) > (d) > (f) > (e)
- Routing: (b) > (e) > (c) > (d) > (f)

#### 2. GNN Depth (3-4 Layers Optimal)

**Placement (View b) - GNN Layers Impact**:
```
GNN Layers | GINE R²   | ResGatedGCN R²
-----------|------------|----------------
3          | 0.8851     | 0.8885
4          | 0.8915     | 0.8815
5          | 0.8853     | 0.8797
6          | 0.8348     | 0.8766
```
4 layers optimal, 6 layers shows degradation (over-smoothing).

**Routing (View b) - GNN Layers Impact**:
```
GNN Layers | GINE R²   | ResGatedGCN R²
-----------|------------|----------------
3          | 0.8759     | 0.8762
4          | 0.8664     | 0.8653
5          | 0.8694     | 0.8742
6          | 0.8708     | 0.8566
```
3 layers optimal.

#### 3. Head Depth (3-4 Layers Critical)

**Placement (View b) - Head Layers Impact**:
```
Head Layers | GINE R²    | ResGatedGCN R²
------------|------------|-----------------
1           | 0.8514     | -0.1689  (failed)
2           | 0.8685     | 0.8832
3           | 0.9883     | 0.9889
4           | 0.9962     | 0.9891
```
3-4 layers significantly improve, ResGatedGCN fails at Head=1.

**Routing (View b) - Head Layers Impact**:
```
Head Layers | GINE R²    | ResGatedGCN R²
------------|------------|-----------------
1           | NaN        | NaN        (all failed)
2           | 0.8586     | 0.8620
3           | 0.9891     | 0.9841
4           | 0.9965     | 0.9841
```
Head=1 all failed, must use 3-4 layers.

#### 4. Model-View Interaction

| Model | Best View (Placement) | Best View (Routing) |
|-------|---------------------|-------------------|
| GINE | (b) R²=0.8878 | (b) R²=0.8596 |
| ResGatedGCN | (b) R²=0.8803 | (b) R²=0.8763 |
| GAT | (b) R²=0.8049 | (e) R²=0.8389 |

---

## 📈 Performance Summary

### Placement (View b, Test R²)

| Model | Head=3 | Head=4 |
|-------|--------|--------|
| GINE | 0.9883 | **0.9962** |
| ResGatedGCN | 0.9889 | 0.9891 |

### Routing (View b, Test R²)

| Model | Head=3 | Head=4 |
|-------|--------|--------|
| GINE | 0.9891 | **0.9965** |
| ResGatedGCN | 0.9841 | 0.9841 |

**Key Finding**: Head depth (3-4 layers) significantly improves accuracy.

---

## 📋 Recommended Configurations

Based on paper ablation experiments:

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| `--model` | `resgatedgcn` | Most stable, consistent across views |
| `--num_gnn_layers` | `3-4` | Message passing depth |
| `--num_head_layers` | `3-4` | Decoder depth, **critical parameter** |
| `--hid_dim` | `256` | Hidden layer dimension |
| `--lr` | `0.0001` | Learning rate |
| `--epochs` | `100` | Training epochs |
| `--lr_scheduler` | `plateau` | Adaptive learning rate adjustment |

### Best Performance Configuration

```bash
# Placement (View b, Head=3)
python main.py \
    --dataset place_B_homograph \
    --task_level node \
    --model resgatedgcn \
    --hid_dim 256 \
    --num_gnn_layers 4 \
    --num_head_layers 3 \
    --lr 0.0001 \
    --epochs 100 \
    --gpu 0

# Routing (View b, Head=3)
python main.py \
    --dataset route_B_homograph \
    --task_level edge \
    --model gine \
    --hid_dim 256 \
    --num_gnn_layers 3 \
    --num_head_layers 3 \
    --lr 0.0001 \
    --epochs 100 \
    --gpu 0
```

---

## ❓ Common Questions

**Q1: Why does training fail when Head=1?**  
A: Decoder head capacity is insufficient to learn complex mapping. Routing task is particularly sensitive, requiring 3-4 layers.

**Q2: Why does ResGatedGCN show negative R² when Head=1?**  
A: Model is severely underfitted. Negative R² indicates prediction variance is less than true value variance.

**Q3: Is deeper GNN always better?**  
A: No. 3-4 layers optimal, deeper (6+) leads to over-smoothing.

**Q4: Which model is best?**  
A: GINE (best overall), ResGatedGCN (most stable), GAT (specialized for view e).

**Q5: How to choose View?**  
A: Prioritize View (b) for both placement and routing.

**Q6: What does label = -1 mean?**  
A: -1 is a sentinel value for invalid/no-label samples. These nodes are ignored during training and evaluation.

**Q7: Why use neighbor sampling?**  
A: Large circuit graphs have millions of nodes/edges. Full-batch training is infeasible. Neighbor sampling creates manageable subgraphs for efficient mini-batch training.

---

**Note**: This is the official reproduction code for CVPR 2026 paper.
