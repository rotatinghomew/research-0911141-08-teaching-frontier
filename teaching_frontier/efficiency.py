"""相对效率与改进空间分析（投入导向 DEA 包络模型）。

这里的效率是相对于所选比较组内其他中心的前沿距离，不是绝对绩效，
也不产生排名或管理责任认定。并列三类模型，避免单一比值包办判断：

- CRS（恒定规模报酬）：包含规模因素的总体相对效率；
- VRS（可变规模报酬）：剔除规模差异后的纯技术相对效率；
- Banker–Morey 调整 VRS：可调整投入允许缩减，不可控条件
  （nondiscretionary）只能要求参照对象不超过本地现状。

质量比例不充当产出，而是参照对象的资格门槛；参照组合是数学上的
线性参照，不代表现场可执行的缩编方案。缺报字段在比较组内整列剔除，
真实零值保留，产出缺报或全部产出为零的中心不进入前沿。

求解前对每一列做常数缩放，因此更换计量单位不会改变效率判断。
"""

from .lp import solve_lp, LinearProgramError

EPS = 1e-9
PEER_WEIGHT_EPS = 1e-6
SCORE_CLIP = 1e-9
QUALITY_GATE_LABEL = "质量门槛"
DEFAULT_QUALITY_THRESHOLD = 0.0
MIN_FRONTIER_SIZE = 6

FIELD_LABELS = {
    "staff_hours": "教师工时",
    "equipment_hours": "设备机时",
    "funding": "经费",
    "space": "场地面积",
    "completed": "完成教学量",
    "complex_modules": "复杂模块数",
}
RTS_LABELS = {"constant": "规模报酬不变（落在规模有效位置）",
              "increasing": "规模报酬递增（偏小，规模因素在压低效率）",
              "decreasing": "规模报酬递减（偏大，规模因素在压低效率）",
              "unknown": "规模状态无法判定"}


def _round(value, digits=6):
    if value is None:
        return None
    rounded = round(float(value), digits)
    if abs(rounded) < EPS:
        rounded = 0.0
    return rounded


def _number(record, field):
    value = record.get(field)
    if value is None:
        return None
    return float(value)


def parse_selection(selection, data):
    """从页面选择中解析比较组与质量要求，未知取值回退到产品默认。"""
    selection = selection if isinstance(selection, dict) else {}
    family = selection.get("family", "all")
    if family != "all" and family not in {row["family"] for row in data["centers"]}:
        family = "all"
    constraint = selection.get("constraint", "all")
    if constraint != "all" and constraint not in {row.get("constraint") for row in data["centers"]}:
        constraint = "all"
    threshold = selection.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_QUALITY_THRESHOLD
    threshold = float(threshold)
    if threshold < 0:
        threshold = 0.0
    if threshold > 1:
        threshold = 1.0
    return {"family": family, "constraint": constraint, "quality_threshold": threshold}


def _field_sets(data):
    return list(data["inputs"]), list(data["outputs"]), list(data["nondiscretionary"])


def _eligible_reference(record, inputs, outputs, nondiscretionary, threshold):
    """返回 (是否可作参照, 原因)。

    投入导向径向模型要求参照对象在每个产出维度都为正值：任一产出为
    零的点会让 theta 收缩为零而仍然满足产出约束，数学上退化。零产出
    中心自身仍可用其他中心作参照接受评估，只是不能进入参照前沿。
    """
    observed_outputs = [_number(record, field) for field in outputs]
    if any(value is None for value in observed_outputs):
        return False, "产出缺报，无法构成完整教学量前沿"
    if any(value is not None and value <= EPS for value in observed_outputs):
        zero_fields = [FIELD_LABELS.get(field, field)
                       for field, value in zip(outputs, observed_outputs) if value is not None and value <= EPS]
        return False, "产出维度 " + "、".join(zero_fields) + " 为真实零，作为参照会使径向效率退化，仅可接受评估"
    quality = _number(record, "quality")
    if threshold > 0 and (quality is None or quality + EPS < threshold):
        observed = "缺报" if quality is None else f"{quality:.3f}"
        return False, f"质量为{observed}，未达到当前{QUALITY_GATE_LABEL} {threshold:g}"
    if any(_number(record, field) is None for field in nondiscretionary):
        return False, "不可控条件缺报，无法约束参照环境"
    return True, ""


def _has_positive_outputs(record, outputs):
    values = [_number(record, field) for field in outputs]
    return any(value is not None and value > EPS for value in values)


def analyze_efficiency(data, raw_selection):
    selection = parse_selection(raw_selection, data)
    inputs, outputs, nondiscretionary = _field_sets(data)
    family, constraint, threshold = selection["family"], selection["constraint"], selection["quality_threshold"]

    in_scope = [row for row in data["centers"]
                if (family == "all" or row["family"] == family)
                and (constraint == "all" or row.get("constraint") == constraint)]
    if not in_scope:
        raise ValueError("此课程类别没有中心")

    reference_flags = {row["center_id"]: _eligible_reference(row, inputs, outputs, nondiscretionary, threshold)
                       for row in in_scope}
    candidates = [row for row in in_scope if reference_flags[row["center_id"]][0]]
    zero_output_references = [row["center_id"] for row in in_scope
                              if row not in candidates
                              and reference_flags[row["center_id"]][1] and "真实零" in reference_flags[row["center_id"]][1]]

    # 缺报的可调整投入在比较组内整列剔除，保证所有参照对象口径一致
    active_inputs = [field for field in inputs
                     if all(_number(row, field) is not None for row in candidates)]
    excluded_inputs = [field for field in inputs if field not in active_inputs]
    units = data.get("units", {})

    boundaries = []
    if not candidates:
        boundaries.append("当前比较组没有满足资格条件的参照对象，所有中心均无法评估相对效率。")
    elif len(candidates) < MIN_FRONTIER_SIZE:
        boundaries.append(f"满足条件的参照对象仅{len(candidates)}个，少于建议的{MIN_FRONTIER_SIZE}个，"
                          "前沿区分度有限，结论只适合小范围讨论。")
    if zero_output_references:
        boundaries.append("中心 " + "、".join(zero_output_references) +
                          " 存在真实零产出维度，保留原始零值但不进入参照前沿，避免径向效率被零值扭曲；其自身仍接受评估。")
    for field in excluded_inputs:
        boundaries.append(f"{FIELD_LABELS.get(field, field)}存在缺报，已在本次比较组内整体剔除，不做插补。")
    if threshold > 0:
        boundaries.append(f"质量门槛为{threshold:g}：未达标或质量缺报的中心不能作为参照对象，但仍以自身数据接受评估。")
    else:
        boundaries.append("质量门槛为0即不设门槛；质量比例仅作资格条件，不计入产出。")

    scales = _column_scales(candidates, active_inputs, outputs, nondiscretionary)
    baseline = None
    if threshold > 0:
        baseline_candidates = [row for row in in_scope
                               if _eligible_reference(row, inputs, outputs, nondiscretionary, 0.0)[0]]
        baseline_inputs = [field for field in inputs
                           if all(_number(row, field) is not None for row in baseline_candidates)]
        if baseline_candidates and len(baseline_candidates) >= 2:
            baseline_scales = _column_scales(baseline_candidates, baseline_inputs, outputs, nondiscretionary)
            baseline = {row["center_id"]: _evaluate(row, baseline_candidates,
                                                     baseline_inputs, outputs, nondiscretionary,
                                                     baseline_scales, units)
                        for row in in_scope}

    results = []
    for row in in_scope:
        center_id = row["center_id"]
        eligible, reason = reference_flags[center_id]
        result = {"center_id": center_id, "family": row["family"],
                  "constraint": row.get("constraint"), "quality": row.get("quality"),
                  "reference_eligible": eligible, "reference_reason": reason or None,
                  "status": "not_evaluated", "score_crs": None, "score_vrs": None,
                  "scale_efficiency": None, "score_adjusted_vrs": None,
                  "returns_to_scale": None, "peers": [], "input_gaps": [],
                  "reference_access": None,
                  "output_targets": [], "stability": None}
        missing_output = any(_number(row, field) is None for field in outputs)
        if not candidates:
            result["status"] = "no_comparable"
            result["status_reason"] = "比较组内没有满足资格条件的参照对象"
        elif not active_inputs:
            result["status"] = "no_comparable"
            result["status_reason"] = "可调整投入全部缺报，无法构造投入导向前沿"
        elif missing_output:
            result["status"] = "no_comparable"
            result["status_reason"] = "本中心存在产出缺报，无法构造完整的产出约束，不做插补评估"
        else:
            evaluation = _evaluate(row, candidates, active_inputs, outputs, nondiscretionary,
                                   scales, units)
            if evaluation is None:
                result["status"] = "no_comparable"
                if not _has_positive_outputs(row, outputs):
                    result["status_reason"] = "本中心教学量为真实零，且参照对象均须具备正产出，投入导向径向模型不适用"
                else:
                    result["status_reason"] = "现有参照对象无法同时覆盖本地不可控条件，缺少可比对象"
            else:
                result.update(evaluation)
                result["status"] = "evaluated"
                result["status_reason"] = None
        if baseline is not None:
            base = baseline.get(center_id)
            current_vrs = result.get("score_vrs")
            if base is not None and base["status"] == "evaluated" and current_vrs is not None:
                base_peers = {peer["center_id"] for peer in base["peers"]}
                current_peers = {peer["center_id"] for peer in result["peers"]}
                result["stability"] = {
                    "baseline_quality_threshold": 0.0,
                    "baseline_score_vrs": _round(base["score_vrs"]),
                    "score_change_vrs": _round(current_vrs - base["score_vrs"]),
                    "baseline_peer_count": len(base_peers),
                    "peers_kept": sorted(base_peers & current_peers),
                    "peers_lost": sorted(base_peers - current_peers),
                    "peers_added": sorted(current_peers - base_peers),
                    "stable": abs(current_vrs - base["score_vrs"]) <= 0.02
                              and not (base_peers ^ current_peers),
                }
        results.append(result)

    unreachable_centers = [item["center_id"] for item in results
                           if item.get("reference_access") and not item["reference_access"]["reachable"]]
    if unreachable_centers:
        boundaries.append("中心 " + "、".join(unreachable_centers) +
                          " 的参照组合在部分投入上高于当前水平（详见各行参照可达性），"
                          "属于门槛下参照规模更大的数学结果，不作为扩编或缩编建议。")
    no_comparable = [item["center_id"] for item in results if item["status"] == "no_comparable"]
    if no_comparable:
        reasons = {}
        for item in results:
            if item["status"] == "no_comparable":
                reasons.setdefault(item["status_reason"], []).append(item["center_id"])
        for reason, identifiers in reasons.items():
            boundaries.append("缺少可比对象的中心：" + "、".join(identifiers) + "。原因：" + reason)

    summary = {
        "比较中心": len(in_scope),
        "可参照对象": len(candidates),
        "已评估中心": sum(1 for item in results if item["status"] == "evaluated"),
        "缺少可比对象": sum(1 for item in results if item["status"] == "no_comparable"),
        "未参与前沿": sum(1 for item in results if not item["reference_eligible"]),
    }
    notes = [
        "效率得分是所选比较组内的相对前沿距离，不构成绩效排名，也不认定任何中心管理存在问题。",
        "参照组合是数学上的线性参照对象，用于解释投入差距；不代表现场可直接执行的缩编方案。",
        "CRS得分包含规模因素，VRS得分剔除规模差异，规模效率=CRS/VRS；调整VRS只允许缩减可调整投入。",
        "求解对每项指标做常数缩放，更换计量单位（如经费由元改为万元）不会改变效率判断与参照组合。",
        "真实零产出保留在原始记录中；产出缺报或含零产出维度的中心不进入参照前沿，缺报投入按比较组整列剔除。",
        "质量是参照资格门槛而非产出；门槛提高会缩小参照集合，请同时查看稳定性信息。",
    ]
    return {
        "method": {"orientation": "投入导向",
                   "models": ["CRS（含规模因素）", "VRS（剔除规模差异）",
                              "Banker-Morey调整VRS（不可控条件不参与缩减）"],
                   "active_inputs": active_inputs,
                   "excluded_inputs": excluded_inputs,
                   "outputs": outputs,
                   "nondiscretionary": nondiscretionary,
                   "field_labels": FIELD_LABELS,
                   "units": units},
        "comparison": {"family": family, "constraint": constraint,
                       "quality_threshold": threshold},
        "summary": summary,
        "boundaries": boundaries,
        "results": results,
        "stability_note": ("门槛为0时无更高参照集合可对比；提高质量门槛后此处显示得分与参照组合的变化。"
                           if threshold == 0 else
                           "稳定性比较以门槛0（不设门槛）的同组结果为基准，查看收紧质量要求后判断是否变化。"),
        "notes": notes,
    }


def _column_scales(candidates, active_inputs, outputs, nondiscretionary):
    scales = {}
    for field in active_inputs + outputs + nondiscretionary:
        values = [_number(row, field) for row in candidates if _number(row, field) is not None]
        maximum = max((abs(value) for value in values), default=1.0)
        scales[field] = maximum if maximum > EPS else 1.0
    return scales


def _evaluate(record, candidates, active_inputs, outputs, nondiscretionary, scales, units):
    vrs = _solve_model(record, candidates, active_inputs, outputs, nondiscretionary,
                       scales, convexity=True, control_nondiscretionary=False)
    if vrs is None:
        return None
    crs = _solve_model(record, candidates, active_inputs, outputs, nondiscretionary,
                       scales, convexity=False, control_nondiscretionary=False)
    adjusted = _solve_model(record, candidates, active_inputs, outputs, nondiscretionary,
                            scales, convexity=True, control_nondiscretionary=True)
    score_vrs = _round(min(1.0, max(SCORE_CLIP, vrs["theta"])))
    score_crs = _round(min(1.0, max(SCORE_CLIP, crs["theta"]))) if crs else None
    scale_efficiency = _round(min(1.0, score_crs / score_vrs)) if score_crs else None
    score_adjusted = _round(min(1.0, max(SCORE_CLIP, adjusted["theta"]))) if adjusted else None

    rts = "unknown"
    if crs:
        lambda_sum = sum(crs["lambdas"])
        if abs(score_crs - score_vrs) <= 1e-5:
            rts = "constant"
        elif lambda_sum < 1 - 1e-5:
            rts = "increasing"
        elif lambda_sum > 1 + 1e-5:
            rts = "decreasing"

    peers = [{"center_id": candidates[index]["center_id"],
              "weight": _round(weight),
              "family": candidates[index]["family"],
              "constraint": candidates[index].get("constraint")}
             for index, weight in enumerate(vrs["lambdas"]) if weight > PEER_WEIGHT_EPS]

    input_gaps = _input_gaps(record, vrs, active_inputs, nondiscretionary, adjusted, units)
    unreachable_fields = [gap["label"] for gap in input_gaps
                          if gap.get("reference_unreachable")]
    reference_access = {
        "reachable": not unreachable_fields,
        "unreachable_fields": unreachable_fields,
        "message": ("参照组合在 " + "、".join(unreachable_fields) +
                    " 上的投入高于本中心当前水平：这是质量门槛下参照规模更大的数学结果，"
                    "不是可执行的扩编或缩编建议，请结合稳定性与原始数据讨论。")
        if unreachable_fields else "参照组合的各项可调整投入均不高于本中心当前水平，差距可作为讨论口径。",
    }
    output_targets = []
    for field in outputs:
        current = _number(record, field)
        if current is None:
            output_targets.append({"field": field, "label": FIELD_LABELS.get(field, field),
                                   "unit": units.get(field, ""), "current": None,
                                   "target": None, "slack": None})
            continue
        target = vrs["targets"].get(field)
        output_targets.append({"field": field, "label": FIELD_LABELS.get(field, field),
                               "unit": units.get(field, ""), "current": _round(current),
                               "target": _round(target),
                               "slack": _round(max(0.0, target - current))})

    return {"status": "evaluated",
            "score_crs": score_crs, "score_vrs": score_vrs,
            "scale_efficiency": scale_efficiency,
            "score_adjusted_vrs": score_adjusted,
            "returns_to_scale": rts, "returns_to_scale_text": RTS_LABELS[rts],
            "peers": peers, "input_gaps": input_gaps,
            "reference_access": reference_access,
            "output_targets": output_targets}


def _input_gaps(record, vrs, active_inputs, nondiscretionary, adjusted, units):
    gaps = []
    for field in active_inputs:
        current = _number(record, field)
        target = vrs["targets"].get(field)
        slack = vrs["slacks"].get(field, 0.0)
        adjusted_target = adjusted["targets"].get(field) if adjusted else None
        # 目标高于当前投入不是“可核减空间”，而是当前门槛下参照组合规模更大、
        # 无法在不增加投入的情况下复现，需要显式标记，避免被误读为缩编建议。
        reference_unreachable = target is not None and current is not None and target > current + EPS
        gaps.append({
            "field": field, "label": FIELD_LABELS.get(field, field),
            "unit": units.get(field, ""),
            "discretionary": True,
            "current": _round(current),
            "target_vrs": _round(target),
            "slack": _round(slack),
            "gap_ratio": _round((current - target) / current) if current and current > EPS else None,
            "reference_unreachable": reference_unreachable,
            "target_adjusted_vrs": _round(adjusted_target),
            "adjusted_gap_ratio": _round((current - adjusted_target) / current)
            if adjusted_target is not None and current and current > EPS else None,
        })
    for field in nondiscretionary:
        current = _number(record, field)
        reference_level = vrs["targets"].get(field)
        gaps.append({
            "field": field, "label": FIELD_LABELS.get(field, field),
            "unit": units.get(field, ""),
            "discretionary": False,
            "current": _round(current),
            "reference_level": _round(reference_level),
            "headroom": _round((current - reference_level) if (current is not None and reference_level is not None) else None),
            "note": "本地不可调整，仅核对参照组合是否处于相同或更紧的条件",
        })
    return gaps


def _solve_model(record, candidates, active_inputs, outputs, nondiscretionary,
                 scales, convexity, control_nondiscretionary):
    """投入导向包络模型；返回 theta、参照权重、目标与松弛，不可行返回 None。"""
    n = len(candidates)
    # 变量顺序：theta, lambda_1..lambda_n, 各输入/产出松弛
    slack_fields = list(active_inputs) + list(outputs)
    objective = [1.0] + [0.0] * (n + len(slack_fields))

    def scaled(center, field):
        value = _number(center, field)
        return 0.0 if value is None else value / scales[field]

    constraints = []
    # 可调整投入标准式：sum lambda x + s_in - theta*x0 = 0
    # 即 sum lambda x <= theta*x0，松弛 s_in 表示参照组合之外仍可核减的投入冗余
    for field in active_inputs:
        coefficients = [-scaled(record, field)] + [scaled(row, field) for row in candidates] \
                       + [0.0] * len(slack_fields)
        slack_index = n + 1 + slack_fields.index(field)
        coefficients[slack_index] = 1.0
        constraints.append((coefficients, "=", 0.0))
    # 产出标准式：sum lambda y - s_out = y0，即参照组合产出不低于当前产出
    for field in outputs:
        coefficients = [0.0] + [scaled(row, field) for row in candidates] + [0.0] * len(slack_fields)
        slack_index = n + 1 + slack_fields.index(field)
        coefficients[slack_index] = -1.0
        constraints.append((coefficients, "=", scaled(record, field)))
    # 不可控条件：Banker-Morey 模型中 theta 只缩放可调整投入，
    # 不可控投入约束不带 theta；普通模型要求参照组合与本地条件相同。
    for field in nondiscretionary:
        current = scaled(record, field)
        coefficients = [0.0] + [scaled(row, field) for row in candidates] \
                       + [0.0] * len(slack_fields)
        if control_nondiscretionary:
            constraints.append((coefficients, "<=", current))
        else:
            constraints.append((coefficients, "=", current))
    if convexity:
        constraints.append(([0.0] + [1.0] * n + [0.0] * len(slack_fields), "=", 1.0))

    try:
        first = solve_lp(objective, constraints)
    except LinearProgramError:
        return None
    if first["status"] != "optimal":
        return None
    theta = first["x"][0]
    lambdas = first["x"][1:1 + n]

    # 第二阶段：固定 theta，最大化全部松弛之和以识别完整改进空间
    second_objective = [0.0] * (1 + n) + [-1.0] * len(slack_fields)
    fixed = [0.0] + [0.0] * n + [0.0] * len(slack_fields)
    fixed[0] = 1.0
    constraints.append((fixed, "=", theta))
    try:
        second = solve_lp(second_objective, constraints)
    except LinearProgramError:
        second = first
    if second["status"] != "optimal":
        second = first
    values = second["x"]
    slacks_scaled = {field: values[n + 1 + slack_fields.index(field)] for field in slack_fields}

    targets, slacks = {}, {}
    for field in active_inputs:
        # 等式约束为 参照投入 - 松弛 = theta*x0，故目标投入即 theta*x0
        targets[field] = theta * _number(record, field)
        slacks[field] = slacks_scaled[field] * scales[field]
    for field in outputs:
        # 等式约束为 参照产出 - 松弛 = y0，故参照目标为 y0+松弛
        targets[field] = _number(record, field) + slacks_scaled[field] * scales[field]
    for field in nondiscretionary:
        targets[field] = sum(weight * _number(row, field)
                             for weight, row in zip(lambdas, candidates))
    return {"theta": theta, "lambdas": lambdas, "targets": targets, "slacks": slacks}
