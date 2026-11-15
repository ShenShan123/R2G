# R2G_multiview_v3_dataset

完整数据为30个设计族×10版本，保留floorplan、placement、cts、route四阶段与B/C/D/E/F五视图，共6000个原生PyG图，原始体积约155GiB。`metadata/`包含轻量数据目录和读取接口。

数据不作为155GiB普通Git文件提交；完整内容压缩为小于2GiB的分卷附件，保存在本仓库对应Release。`dataset_release_manifest.json`记录每卷字节数和SHA256，以及拼接压缩流的SHA256。所有分卷均下载校验成功后方可解压。

```bash
# 从顶层R2G300_Multiview_Benchmark目录执行
python dataset/download_dataset.py --extract
```

默认在本目录下载分卷，并解压为`dataset/R2G_multiview_v3_dataset/`。可用`--output-dir /path/with/space`指定其他存储路径。磁盘需容纳约155GiB解压数据以及压缩分卷；已校验的分卷会复用。不带`--extract`只下载并校验。

图里的原始绝对路径只作为历史provenance，不作为读取时的必要路径。按manifest相对路径读取，不把布线后标签作为输入。实验中的placement实际输入截止点是floorplan，route标签跨阶段共享。
