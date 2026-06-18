"""
多目标进化搜索 POLY_SCHEMES 的 Pareto 前沿（替代 ILP 固定预算单解）。

目标（均越小越好）：
  f_loss = Σ_j S_{j,k_j}   （敏感度得分和，来自 CSV）
  f_cost = C_bts(scheme)   （完整方案 bootstrap 次数；未实现前用 depth 和占位）

支配关系（x1 支配 x2）：
  x1 在所有目标上都不比 x2 差，且在至少一个目标上严格更好。

  loss：严格更好当 L(x1) < 0.9 * L(x2)；不比 x2 差当 L(x1) <= 1.1 * L(x2)
  cost（depth 占位）：严格更好当差值 > 10（即 C(x1) <= C(x2) - 11）；
                      不比 x2 差当 C(x1) <= C(x2) + 10
  cost（bts 实现后，--cost-mode bts）：严格整数比较，小者优，无容差

非法个体：交叉在合法父代下不产生非法解；初始化/变异若非法则重生成。
Archive：维护至今发现的非支配方案集合，作为 Pareto 前沿输出。
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass, field

from cost import NUM_LAYERS, compute_scheme_cost, scheme_index, validate_scheme
from gelu_poly import gelu_level_allowed

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]
SENSITIVE_OUTPUT_DIR = "./results/sensitive_scores_1/"
POLY_LEVELS = (0, 1, 2)
OUTPUT_DIR = "./results/evolution_results/"

POPULATION_SIZE = 120
NUM_GENERATIONS = 300
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / (NUM_LAYERS * 2)
TOURNAMENT_SIZE = 2
MAX_REGEN_ATTEMPTS = 32
ARCHIVE_MAX_SIZE = 80
RANDOM_SEED = 42

LOSS_STRICT_RATIO = 0.9
LOSS_NOT_WORSE_RATIO = 1.1
COST_STRICT_MARGIN = 11
COST_NOT_WORSE_MARGIN = 10

RUN_INFERENCE_ON_ARCHIVE = False
INFERENCE_ARCHIVE_TOP_K = 5
# ==================================================


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in ("softmax", "gelu")
    ]


def load_sensitivity_matrix(
    task_name: str,
    sensitive_dir: str = SENSITIVE_OUTPUT_DIR,
) -> dict[tuple[int, str], dict[int, float]]:
    path = os.path.join(sensitive_dir, f"{task_name}_sensitivity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"敏感度文件不存在：{path}")

    out: dict[tuple[int, str], dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer_idx = int(row["layer_idx"])
            kind = row["kind"]
            out[(layer_idx, kind)] = {
                0: float(row["S_low"]),
                1: float(row["S_mid"]),
                2: float(row["S_high"]),
            }
    if len(out) != NUM_LAYERS * 2:
        raise ValueError(f"{task_name} 敏感度行数应为 24，当前为 {len(out)}")
    return out


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "softmax":
        return POLY_LEVELS
    return tuple(
        level for level in POLY_LEVELS if gelu_level_allowed(layer_idx, level)
    )


def build_allowed_levels_table() -> list[tuple[int, ...]]:
    table: list[tuple[int, ...]] = []
    for layer_idx, kind in iter_positions():
        table.append(allowed_levels(layer_idx, kind))
    return table


def is_scheme_legal(scheme: list[int], allowed_table: list[tuple[int, ...]]) -> bool:
    if len(scheme) != NUM_LAYERS * 2:
        return False
    for idx, level in enumerate(scheme):
        if level not in allowed_table[idx]:
            return False
    return True


def random_legal_scheme(rng: random.Random, allowed_table: list[tuple[int, ...]]) -> list[int]:
    return [rng.choice(allowed_table[idx]) for idx in range(NUM_LAYERS * 2)]


def random_scheme_with_mutation_draw(
    rng: random.Random, allowed_table: list[tuple[int, ...]]
) -> list[int]:
    """随机 scheme，某位可能抽到非法档位（用于变异路径）。"""
    scheme = random_legal_scheme(rng, allowed_table)
    idx = rng.randrange(NUM_LAYERS * 2)
    scheme[idx] = rng.choice(POLY_LEVELS)
    return scheme


def regenerate_until_legal(
    rng: random.Random,
    allowed_table: list[tuple[int, ...]],
    *,
    prefer_mutated: bool = False,
) -> list[int]:
    for _ in range(MAX_REGEN_ATTEMPTS):
        scheme = (
            random_scheme_with_mutation_draw(rng, allowed_table)
            if prefer_mutated
            else random_legal_scheme(rng, allowed_table)
        )
        if is_scheme_legal(scheme, allowed_table):
            return scheme
    return random_legal_scheme(rng, allowed_table)


def init_population_strategy(
    rng: random.Random, allowed_table: list[tuple[int, ...]], n: int
) -> list[list[int]]:
    population: list[list[int]] = []
    n_all_high = n * 3 // 10
    n_all_low = n * 3 // 10

    for _ in range(n_all_high):
        scheme = []
        for idx, levels in enumerate(allowed_table):
            scheme.append(max(levels))
        population.append(scheme)

    for _ in range(n_all_low):
        scheme = []
        for idx, levels in enumerate(allowed_table):
            scheme.append(min(levels))
        population.append(scheme)

    while len(population) < n:
        population.append(random_legal_scheme(rng, allowed_table))

    rng.shuffle(population)
    return population


def compute_f_loss(
    scheme: list[int],
    s_mat: dict[tuple[int, str], dict[int, float]],
) -> float:
    total = 0.0
    for layer_idx, kind in iter_positions():
        level = scheme[scheme_index(layer_idx, kind)]
        total += s_mat[(layer_idx, kind)][level]
    return total


def compute_f_cost(task_name: str, scheme: list[int], *, cost_mode: str) -> int:
    """
    f_cost = C_bts(scheme)。

    cost_mode='depth'：占位，用乘法深度之和。
    cost_mode='bts'  ：预留，当前仍回退 depth，待 bts 求解器接入。
    """
    validate_scheme(scheme, task_name)
    if cost_mode == "bts":
        # TODO: return int(optimize_bootstrap(task_name, scheme))
        pass
    return int(compute_scheme_cost(task_name, scheme))


@dataclass
class Individual:
    scheme: list[int]
    f_loss: float = 0.0
    f_cost: int = 0
    rank: int = field(default=0, compare=False)
    crowding: float = field(default=0.0, compare=False)


def loss_strictly_better(l1: float, l2: float) -> bool:
    if l2 <= 0.0:
        return l1 < l2
    return l1 < LOSS_STRICT_RATIO * l2


def loss_not_worse(l1: float, l2: float) -> bool:
    if l2 <= 0.0:
        return l1 <= l2
    return l1 <= LOSS_NOT_WORSE_RATIO * l2


def cost_strictly_better(c1: int, c2: int, *, cost_mode: str) -> bool:
    if cost_mode == "bts":
        return c1 < c2
    return c1 <= c2 - COST_STRICT_MARGIN


def cost_not_worse(c1: int, c2: int, *, cost_mode: str) -> bool:
    if cost_mode == "bts":
        return c1 <= c2
    return c1 <= c2 + COST_NOT_WORSE_MARGIN


def dominates(a: Individual, b: Individual, *, cost_mode: str) -> bool:
    if not loss_not_worse(a.f_loss, b.f_loss):
        return False
    if not cost_not_worse(a.f_cost, b.f_cost, cost_mode=cost_mode):
        return False
    loss_better = loss_strictly_better(a.f_loss, b.f_loss)
    cost_better = cost_strictly_better(a.f_cost, b.f_cost, cost_mode=cost_mode)
    return loss_better or cost_better


def evaluate_individual(
    ind: Individual,
    task_name: str,
    s_mat: dict[tuple[int, str], dict[int, float]],
    *,
    cost_mode: str,
) -> Individual:
    ind.f_loss = compute_f_loss(ind.scheme, s_mat)
    ind.f_cost = compute_f_cost(task_name, ind.scheme, cost_mode=cost_mode)
    return ind


def fast_non_dominated_sort(
    population: list[Individual], *, cost_mode: str
) -> list[list[Individual]]:
    n = len(population)
    domination_count = [0] * n
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(population[p], population[q], cost_mode=cost_mode):
                dominated_sets[p].append(q)
            elif dominates(population[q], population[p], cost_mode=cost_mode):
                domination_count[p] += 1
        if domination_count[p] == 0:
            population[p].rank = 1
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return [[population[idx] for idx in front] for front in fronts]


def crowding_distance(front: list[Individual]) -> None:
    if not front:
        return
    for ind in front:
        ind.crowding = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowding = float("inf")
        return

    objectives = (
        ("f_loss", lambda x: x.f_loss),
        ("f_cost", lambda x: float(x.f_cost)),
    )
    for _, getter in objectives:
        front.sort(key=getter)
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        lo = getter(front[0])
        hi = getter(front[-1])
        span = hi - lo
        if span <= 0.0:
            continue
        for i in range(1, len(front) - 1):
            if front[i].crowding == float("inf"):
                continue
            front[i].crowding += (getter(front[i + 1]) - getter(front[i - 1])) / span


def tournament_select(
    rng: random.Random, population: list[Individual], *, cost_mode: str
) -> Individual:
    candidates = rng.sample(population, TOURNAMENT_SIZE)
    best = candidates[0]
    for cand in candidates[1:]:
        if cand.rank < best.rank:
            best = cand
        elif cand.rank == best.rank and cand.crowding > best.crowding:
            best = cand
        elif cand.rank == best.rank and cand.crowding == best.crowding:
            if dominates(cand, best, cost_mode=cost_mode):
                best = cand
    return best


def crossover(
    rng: random.Random,
    parent1: list[int],
    parent2: list[int],
) -> list[int]:
    if rng.random() >= CROSSOVER_RATE:
        return parent1.copy()
    return [
        parent1[i] if rng.random() < 0.5 else parent2[i]
        for i in range(NUM_LAYERS * 2)
    ]


def mutate(
    rng: random.Random,
    scheme: list[int],
    allowed_table: list[tuple[int, ...]],
) -> list[int]:
    child = scheme.copy()
    if rng.random() >= MUTATION_RATE:
        return child
    idx = rng.randrange(NUM_LAYERS * 2)
    child[idx] = rng.choice(POLY_LEVELS)
    if is_scheme_legal(child, allowed_table):
        return child
    return regenerate_until_legal(rng, allowed_table, prefer_mutated=True)


def environmental_selection(
    combined: list[Individual],
    target_size: int,
    *,
    cost_mode: str,
) -> list[Individual]:
    fronts = fast_non_dominated_sort(combined, cost_mode=cost_mode)
    next_pop: list[Individual] = []
    for front in fronts:
        crowding_distance(front)
        if len(next_pop) + len(front) <= target_size:
            next_pop.extend(front)
        else:
            front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_loss))
            need = target_size - len(next_pop)
            next_pop.extend(front[:need])
            break
    return next_pop


class Archive:
    """维护非支配解集合（Pareto 前沿容器）。"""

    def __init__(self, *, cost_mode: str, max_size: int = ARCHIVE_MAX_SIZE) -> None:
        self.cost_mode = cost_mode
        self.max_size = max_size
        self.items: list[Individual] = []

    def _scheme_key(self, scheme: list[int]) -> tuple[int, ...]:
        return tuple(scheme)

    def consider(self, ind: Individual) -> None:
        key = self._scheme_key(ind.scheme)
        for existing in self.items:
            if self._scheme_key(existing.scheme) == key:
                return

        if any(dominates(other, ind, cost_mode=self.cost_mode) for other in self.items):
            return

        self.items = [
            other
            for other in self.items
            if not dominates(ind, other, cost_mode=self.cost_mode)
        ]
        self.items.append(
            Individual(
                scheme=ind.scheme.copy(),
                f_loss=ind.f_loss,
                f_cost=ind.f_cost,
            )
        )
        self._truncate()

    def extend(self, population: list[Individual]) -> None:
        for ind in population:
            self.consider(ind)

    def _truncate(self) -> None:
        if len(self.items) <= self.max_size:
            return
        fronts = fast_non_dominated_sort(self.items, cost_mode=self.cost_mode)
        kept: list[Individual] = []
        for front in fronts:
            crowding_distance(front)
            if len(kept) + len(front) <= self.max_size:
                kept.extend(front)
            else:
                front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_loss))
                kept.extend(front[: self.max_size - len(kept)])
                break
        self.items = kept

    def sorted_items(self) -> list[Individual]:
        return sorted(self.items, key=lambda x: (x.f_cost, x.f_loss))


def run_evolution(
    task_name: str,
    *,
    population_size: int = POPULATION_SIZE,
    num_generations: int = NUM_GENERATIONS,
    cost_mode: str = "depth",
    seed: int = RANDOM_SEED,
    sensitive_dir: str = SENSITIVE_OUTPUT_DIR,
) -> tuple[list[Individual], Archive]:
    rng = random.Random(seed)
    s_mat = load_sensitivity_matrix(task_name, sensitive_dir)
    allowed_table = build_allowed_levels_table()

    population = [
        Individual(scheme=scheme)
        for scheme in init_population_strategy(rng, allowed_table, population_size)
    ]
    for ind in population:
        evaluate_individual(ind, task_name, s_mat, cost_mode=cost_mode)

    archive = Archive(cost_mode=cost_mode)
    archive.extend(population)

    for _gen in range(num_generations):
        offspring: list[Individual] = []
        ranks = fast_non_dominated_sort(population, cost_mode=cost_mode)
        flat = [ind for front in ranks for ind in front]
        for front in ranks:
            crowding_distance(front)

        while len(offspring) < population_size:
            p1 = tournament_select(rng, flat, cost_mode=cost_mode)
            p2 = tournament_select(rng, flat, cost_mode=cost_mode)
            child_scheme = crossover(rng, p1.scheme, p2.scheme)
            child_scheme = mutate(rng, child_scheme, allowed_table)
            child = Individual(scheme=child_scheme)
            evaluate_individual(child, task_name, s_mat, cost_mode=cost_mode)
            offspring.append(child)

        population = environmental_selection(
            population + offspring,
            population_size,
            cost_mode=cost_mode,
        )
        archive.extend(population)

    return archive.sorted_items(), archive


def format_scheme(scheme: list[int]) -> str:
    return str(scheme)


def save_pareto_csv(task_name: str, archive: Archive, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rank_hint", "f_loss", "f_cost", "scheme"])
        for i, ind in enumerate(archive.sorted_items(), start=1):
            writer.writerow(
                [task_name, i, f"{ind.f_loss:.8e}", ind.f_cost, format_scheme(ind.scheme)]
            )


def format_elapsed(seconds: float) -> str:
    """将秒数格式化为易读字符串。"""
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def print_archive(
    task_name: str,
    items: list[Individual],
    *,
    cost_mode: str,
    elapsed_s: float | None = None,
) -> None:
    cost_label = "bts" if cost_mode == "bts" else "depth(占位)"
    print(f"\n{'=' * 72}")
    print(f"Pareto 前沿：{task_name.upper()}  |  cost={cost_label}  |  共 {len(items)} 个解")
    if elapsed_s is not None:
        print(f"搜索用时：{format_elapsed(elapsed_s)} ({elapsed_s:.2f}s)")
    print(f"{'=' * 72}")
    print(f"{'#':<4} {'f_loss':<14} {'f_cost':<8} scheme")
    print("-" * 72)
    for i, ind in enumerate(items, start=1):
        print(f"{i:<4} {ind.f_loss:<14.6e} {ind.f_cost:<8} {format_scheme(ind.scheme)}")


def maybe_run_inference(task_name: str, items: list[Individual], top_k: int) -> None:
    if not items:
        return
    from poly_model_inference import (
        fmt_accuracy,
        fmt_accuracy_delta,
        run_task_with_comparison,
    )

    print(f"\n>>> Archive 前 {top_k} 个方案验证集推理：{task_name.upper()}")
    print(
        f"{'#':<4} {'f_loss':<12} {'f_cost':<8} "
        f"{'poly_acc':<10} {'Δacc':<10} {'poly_loss':<12}"
    )
    print("-" * 70)
    for i, ind in enumerate(items[:top_k], start=1):
        result = run_task_with_comparison(task_name, ind.scheme)
        print(
            f"{i:<4} {ind.f_loss:<12.4e} {ind.f_cost:<8} "
            f"{fmt_accuracy(result['poly_accuracy']):<10} "
            f"{fmt_accuracy_delta(result['accuracy_delta']):<10} "
            f"{result['poly_loss']:.7f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多目标进化搜索 POLY_SCHEMES Pareto 前沿（ΣS vs C_bts）"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASK_NAMES,
        help="任务列表（默认 mrpc rte sst2）",
    )
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE, help="种群规模")
    parser.add_argument("--gens", type=int, default=NUM_GENERATIONS, help="进化代数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    parser.add_argument(
        "--cost-mode",
        choices=("depth", "bts"),
        default="depth",
        help="depth=深度和占位+bts容差; bts=严格整数比较（bts 未接入时仍用 depth 值）",
    )
    parser.add_argument(
        "--sensitive-dir",
        default=SENSITIVE_OUTPUT_DIR,
        help="敏感度 CSV 目录",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Pareto CSV 输出目录",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        default=RUN_INFERENCE_ON_ARCHIVE,
        help="对 Archive 前 K 个方案跑验证集推理",
    )
    parser.add_argument(
        "--inference-top-k",
        type=int,
        default=INFERENCE_ARCHIVE_TOP_K,
        help="推理评估的 Archive 方案数",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("多目标进化搜索（支配：loss 10% 容差；cost depth 占位 ±10 / bts 严格）")
    print(
        f"pop={args.pop}, gens={args.gens}, seed={args.seed}, "
        f"cost_mode={args.cost_mode}, sensitive_dir={args.sensitive_dir}"
    )

    task_timings: list[tuple[str, float]] = []

    for task_name in args.tasks:
        try:
            t0 = time.perf_counter()
            items, archive = run_evolution(
                task_name,
                population_size=args.pop,
                num_generations=args.gens,
                cost_mode=args.cost_mode,
                seed=args.seed,
                sensitive_dir=args.sensitive_dir,
            )
            elapsed_s = time.perf_counter() - t0
        except FileNotFoundError as exc:
            print(f"\n{task_name.upper()} 跳过：{exc}")
            continue

        print_archive(
            task_name, items, cost_mode=args.cost_mode, elapsed_s=elapsed_s
        )
        out_path = os.path.join(args.output_dir, f"{task_name}_pareto.csv")
        save_pareto_csv(task_name, archive, out_path)
        print(f"已保存：{out_path}")
        task_timings.append((task_name, elapsed_s))

        if args.inference:
            maybe_run_inference(task_name, items, args.inference_top_k)

    if task_timings:
        print(f"\n{'=' * 72}")
        print("搜索用时汇总")
        print(f"{'=' * 72}")
        print(f"{'任务':<8} {'用时':<16} {'秒':<10}")
        print("-" * 36)
        total_search_s = 0.0
        for task_name, elapsed_s in task_timings:
            print(
                f"{task_name:<8} {format_elapsed(elapsed_s):<16} {elapsed_s:<10.2f}"
            )
            total_search_s += elapsed_s
        print("-" * 36)
        print(
            f"{'合计':<8} {format_elapsed(total_search_s):<16} {total_search_s:<10.2f}"
        )


if __name__ == "__main__":
    main()
