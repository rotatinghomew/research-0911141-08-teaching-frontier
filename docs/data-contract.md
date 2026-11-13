# 数据口径

centers 包含所属课程 family、固定场地 constraint 和可调整投入。inputs 中 staff_hours、equipment_hours、funding 量纲不同，space 为不可控条件。outputs 中 completed、complex_modules 为教学量，quality 是已测质量比例而非额外产量。null 表示缺报；0 表示真实零。单位换算由 units 描述，不包含预定效率或排名。

## 数据版本

`fixtures/data.json` 是可直接导入的一份完整材料，`product.json` 保存页面默认范围。导入必须提交完整对象，不把省略字段解释为删除补丁；以规范化 JSON 的 SHA-256 标识内容，同内容多次导入复用版本。原始记录只追加版本，保存的分析材料始终指向原来版本。

页面中的范围影响当前描述，不回写源数据。`null`、缺记录与实际零不能合并；数据格式问题应在导入时明确报错，而不是生成看似成功的空结果。涉及单位、时间或关系的解释以本文件为准，统计方法和业务决策没有预设结果。

## 目录

`teaching_frontier/domain.py` 提供现行领域校验和描述，`validation.py` 处理基础引用与数值校验，`service.py` 连接领域操作和材料存档，`storage.py` 负责内容寻址数据与历史记录，`server.py` 提供本地接口。`web/app.js` 使用接口构建交互页面，不自行推导独立口径。
