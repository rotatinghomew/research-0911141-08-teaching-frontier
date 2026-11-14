# 本地接口

服务只用于本地教研复核，不连接真实人员目录或外部数据源。数据以 UTF-8 JSON 传输；未找到对象返回 404，无效业务输入返回 400，正文超出 8MB 返回 413。

- `GET /health` 返回就绪状态。
- `GET /api/context` 返回产品配置、默认版本、已导入版本与保存记录目录。
- `GET /api/data?revision=版本` 返回该版本完整原始数据；省略版本使用演示材料。
- `POST /api/datasets` 接收 `label` 和完整 `data`，返回新建或复用的 `revision`。
- `POST /api/explore` 接收可选 `revision` 和 `selection`，返回数据指纹、实际选择和领域描述。`selection` 支持 `family`（课程类别）、`constraint`（当地约束）和 `quality_threshold`（质量门槛，0 表示不设门槛）；未知取值回退到产品默认，不返回 500。
- `POST /api/notebooks` 使用相同选择参数和 `title`，服务端重新计算后保存，不能由客户端伪造结果。
- `GET /api/notebooks` 返回材料目录，`GET /api/notebooks/编号` 返回冻结描述、选择条件和所用数据。

领域描述在原有 `summary`、`rows`、`series`、`notes` 之外增加 `efficiency` 区块：`method` 列出参与与剔除的投入产出、模型和单位，`comparison` 回显实际选择，`boundaries` 逐条说明缺报、零产出和参照资格的处理边界，`results` 按中心原始顺序给出 CRS/VRS/规模效率/调整 VRS 得分、规模状态、参照组合与权重、每项投入的当前值与目标值（含原始单位）、产出目标和稳定性对比，`stability_note` 与 `notes` 说明解释口径。这里记录的是已有浏览交互，不限制继续扩展分析结构。增添业务能力时需保持现有原始记录和保存材料可读取。
