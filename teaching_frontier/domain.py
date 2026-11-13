from collections import Counter, defaultdict
from statistics import mean, median
import math

from .validation import require_tables, unique, finite, references


def validate(data):
    require_tables(data, ["centers", "inputs", "outputs", "nondiscretionary"])
    unique(data["centers"], ["center_id"])
    for center in data["centers"]:
        for field in data["inputs"] + data["outputs"] + data["nondiscretionary"]:
            if center[field] is not None:
                finite(center[field], minimum=0)
        if center["quality"] is not None:
            finite(center["quality"], minimum=0, maximum=1)


def explore(data, selection):
    family = selection.get("family", "all")
    centers = [record for record in data["centers"] if family == "all" or record["family"] == family]
    if not centers:
        raise ValueError("此课程类别没有中心")
    rows = []
    for center in centers:
        rows.append({"center_id": center["center_id"], "family": center["family"], "quality": center["quality"],
                     "completed": center["completed"], "complex_modules": center["complex_modules"],
                     "completed_per_staff_hour": center["completed"] / center["staff_hours"] if center["staff_hours"] else None,
                     "funding_reported": center["funding"] is not None})
    return {"summary": {"中心": len(rows), "已报资金": sum(row["funding_reported"] for row in rows),
                        "完成教学量": sum(row["completed"] for row in rows)},
            "rows": rows, "series": [{"label": row["center_id"], "y": row["completed_per_staff_hour"]} for row in rows],
            "notes": ["单项比率只作数据核对，不构成多投入产出的效率排序。", "质量比例与产出数量量纲不同，不能直接相加。"]}
