"""仅依赖标准库的两阶段单纯形求解器。

服务于教学资源相对效率分析中的小规模线性规划（变量与约束均为
几十个中心量级）。使用 Bland 规则选基以避免退化循环，所有问题先
转为 b>=0 的标准型。求解器只处理 x>=0 的线性规划，返回最优解、
不可行或无界三种状态。
"""

EPS = 1e-9
PIVOT_EPS = 1e-8
INFEASIBLE_EPS = 1e-7


class LinearProgramError(ValueError):
    """线性规划结构无法求解时抛出。"""


def _pivot(tableau, basis, entering, leaving):
    row = tableau[leaving]
    pivot_value = row[entering]
    if abs(pivot_value) <= PIVOT_EPS:
        raise LinearProgramError("主元数值过小，无法保持数值稳定")
    tableau[leaving] = [value / pivot_value for value in row]
    reduced = tableau[leaving]
    for index, current in enumerate(tableau):
        if index == leaving:
            continue
        factor = current[entering]
        if factor == 0.0:
            continue
        tableau[index] = [value - factor * other for value, other in zip(current, reduced)]
    basis[leaving] = entering


def _bland_step(tableau, basis, objective_row):
    entering = None
    for column in range(len(tableau[0]) - 1):
        if tableau[objective_row][column] < -PIVOT_EPS:
            entering = column
            break
    if entering is None:
        return False
    leaving = None
    best_ratio = None
    for row_index in range(objective_row):
        coefficient = tableau[row_index][entering]
        if coefficient <= PIVOT_EPS:
            continue
        ratio = tableau[row_index][-1] / coefficient
        if ratio < -EPS:
            continue
        if best_ratio is None or ratio < best_ratio - EPS or (
            abs(ratio - best_ratio) <= EPS and basis[row_index] < basis[leaving]
        ):
            best_ratio = ratio
            leaving = row_index
    if leaving is None:
        raise LinearProgramError("线性规划无界")
    _pivot(tableau, basis, entering, leaving)
    return True


def solve_lp(objective, constraints):
    """求解 min c^T x，约束为 (a, sense, rhs) 三元组，变量均满足 x>=0。

    返回 {"status": "optimal"|"infeasible"|"unbounded",
          "x": 原始变量取值, "objective": 目标值}。
    """
    if not objective or any(isinstance(value, bool) for value in objective):
        raise LinearProgramError("目标函数必须是非空数值向量")
    n = len(objective)
    normalized = []
    for coefficients, sense, rhs in constraints:
        if len(coefficients) != n or sense not in ("<=", ">=", "="):
            raise LinearProgramError("约束系数维度或方向不合法")
        if rhs < -EPS:
            coefficients = [-value for value in coefficients]
            rhs = -rhs
            sense = ">=" if sense == "<=" else "<=" if sense == ">=" else "="
        normalized.append((coefficients, sense, max(0.0, rhs)))

    widths = []
    for _, sense, _ in normalized:
        if sense == "<=":
            widths.append(1)          # 松弛变量
        elif sense == ">=":
            widths.append(2)          # 剩余变量 + 人工变量
        else:
            widths.append(1)          # 人工变量
    ncols = n + sum(widths)
    tableau = [[0.0] * (ncols + 1) for _ in range(len(normalized) + 1)]
    basis = [0] * len(normalized)
    artificial = set()
    cursor = n
    for row_index, ((coefficients, sense, rhs), width) in enumerate(zip(normalized, widths)):
        for column, value in enumerate(coefficients):
            tableau[row_index][column] = float(value)
        tableau[row_index][-1] = float(rhs)
        if sense == "<=":
            tableau[row_index][cursor] = 1.0
            basis[row_index] = cursor
            cursor += 1
        elif sense == ">=":
            tableau[row_index][cursor] = -1.0          # 剩余变量
            cursor += 1
            tableau[row_index][cursor] = 1.0           # 人工变量
            artificial.add(cursor)
            basis[row_index] = cursor
            cursor += 1
        else:
            tableau[row_index][cursor] = 1.0
            artificial.add(cursor)
            basis[row_index] = cursor
            cursor += 1

    # 第一阶段：最小化人工变量之和。目标行写简约成本，右端项为目标值相反数
    phase_one = len(tableau) - 1
    for column in artificial:
        tableau[phase_one][column] = 1.0
    for row_index in range(phase_one):
        if basis[row_index] in artificial:
            factor = tableau[phase_one][basis[row_index]]
            for column in range(ncols + 1):
                tableau[phase_one][column] -= factor * tableau[row_index][column]
    while _bland_step(tableau, basis, phase_one):
        pass
    if tableau[phase_one][-1] < -INFEASIBLE_EPS:
        return {"status": "infeasible", "x": [0.0] * n, "objective": None}

    # 清理仍在基中的人工变量，无法换出的行是冗余约束，直接删除
    keep_rows = []
    for row_index in range(phase_one):
        if basis[row_index] not in artificial:
            keep_rows.append(row_index)
            continue
        swapped = False
        for column in range(ncols):
            if column in artificial or abs(tableau[row_index][column]) <= PIVOT_EPS:
                continue
            _pivot(tableau, basis, column, row_index)
            keep_rows.append(row_index)
            swapped = True
            break
        if not swapped:
            basis[row_index] = -1
    constraint_rows = [tableau[index] for index in keep_rows]
    basis = [basis[index] for index in keep_rows]
    for row in constraint_rows:
        for column in artificial:
            row[column] = 0.0

    # 第二阶段：原目标函数的简约成本行，右端项为最小目标值的相反数
    objective_row = [0.0] * (ncols + 1)
    objective_row[:n] = [float(value) for value in objective]
    for row_index, basic in enumerate(basis):
        if basic < 0:
            continue
        factor = objective_row[basic]
        if factor:
            for column in range(ncols + 1):
                objective_row[column] -= factor * constraint_rows[row_index][column]
    tableau = constraint_rows + [objective_row]
    basis.append(-1)
    objective_index = len(tableau) - 1
    try:
        while _bland_step(tableau, basis, objective_index):
            pass
    except LinearProgramError:
        return {"status": "unbounded", "x": [0.0] * n, "objective": None}

    solution = [0.0] * n
    for row_index, basic in enumerate(basis[:-1]):
        if 0 <= basic < n:
            solution[basic] = tableau[row_index][-1]
    return {"status": "optimal", "x": solution, "objective": -tableau[objective_index][-1]}
