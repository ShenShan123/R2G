# Placement拥塞任务、输入坐标与结果范围

核查对象：300实例v3数据、原六模型工程，以及发布包所列历史结果。本次未重新训练或修改数据。

| 问题 | 核对结论 | 证据 |
|---|---|---|
| 所有历史placement都没跑拥塞吗？ | 否。旧L3/H2、seed42批次有6模型×B/C/D/F=24项complete结果。 | baselines/raw中legacy_L3H2_100_seed42__placement__congestion开头的JSON |
| v3 L4/H3已有placement拥塞结果吗？ | 在检查的归档中未找到；v3同批次覆盖记录保留缺失；最终汇总主表使用旧批次已完成结果，并明确标注来源。不能由缺失记录反推当时确切取消原因。 | v3主表与配置记录 |
| 名为placement的v3图使用已布局坐标吗？ | 300/300份B视图metadata的真实输入截止点是floorplan；standard_cell_coordinates=false、gate_gate_geometry=false。宏/IO固定坐标与标准单元坐标需分别看待。 | provenance/placement_congestion_audit.json |
| 缺坐标具体限制什么？ | 无法用未知标准单元位置确定输入阶段的网格归属、构建可信的Gate空间邻接或展示空间拥塞热图。 | feature_availability_contract、auxiliary_edge_availability_contract |
| 是否因此不能预测Gate拥塞？ | 仍可预测。最终route/post-route标签按Gate实例身份对齐到早期图；无需用未知早期坐标重新计算标签。输入仍可使用合法拓扑与单元/全局特征。 | shared_label_contract、task_registry和实际图标签 |
| E视图为什么不跑拥塞？ | 原E视图只有Pin/IO，没有Gate，当前cell_congestion监督目标不适用；这是监督载体问题。 | 六模型dataset.py的TaskSpec |

抽查实际PT：ac97_ctrl/v01的placement B图有10126个Gate，其中10125个拥塞有效标签；route B图同样有10125个有效标签，两者有效标签值范围均0—0.4390333。placement未存储congestion_geom关系，route有29968条此类边。该例只说明标签可跨阶段对齐，不能外推为所有图都逐张量完全相等。合法零拥塞标签仍有效。

缺少可信输入坐标只影响空间信息可用性，不能据此断言拥塞训练没有意义或历史从未运行。也不能使用route真实坐标补回floorplan输入，否则改变预测时点与数据契约。

## 已加入的结果表

[旧批次placement拥塞：六模型×五视图矩阵和24项详细指标](../baselines/tables/PLACEMENT_CONGESTION_LEGACY.md)，并提供[CSV明细](../baselines/placement_congestion_legacy.csv)。批次明确标为legacy_L3H2_100_seed42；v3缺失状态不变。最终Baseline主表与Excel的placement拥塞位置已直接填入这24项结果，标注旧批次。
