"""
多目标进化搜索 POLY_SCHEMES 的 Pareto 前沿（Output KL vs C_bts）。

目标：
  f_output_kl = KL(p_baseline ‖ p_poly)（校验集平均，越小越好；严格比较，无容差）
  f_cost      = ceil(深度和 / COST_DEPTH_DIVISOR)（越小越好；严格比较，无容差）

评估集：进化搜索默认使用全部校验集（calib）；可用 --eval-samples N 抽样子集加速。
        Pareto 解最终分别在全部校验集与全部验证集上汇报，写出两个 CSV。
输出 CSV：精简列（无 task/rank_hint/oor_*）；非有限 KL 搜索期不入 archive。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

from cost import (
    NUM_LAYERS,
    SCHEME_LEN,
    compute_scheme_cost,
    depth_f_cost_label,
    depth_sum_to_f_cost,
    validate_scheme as validate_cost_scheme,
)
from evolution_infer import (
    SchemeEvaluator,
    device,
    enrich_pareto_report,
    fmt_flips_pct,
    format_elapsed,
    save_search_timings,
)
from gelu_poly import gelu_level_allowed
from poly_model_inference import (
    CALIB_INDICES_DIR,
    CALIB_SEED,
    fmt_metric_delta,
)
# ===================== 配置区 =====================
#TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
TASK_NAMES = ["cola", "qnli", "mnli"]
POLY_LEVELS = (0, 1, 2)
OUTPUT_DIR = "./results/evolution_kl_results/"

POPULATION_SIZE = 120
NUM_GENERATIONS = 300
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / SCHEME_LEN
TOURNAMENT_SIZE = 2
MAX_REGEN_ATTEMPTS = 32
ARCHIVE_MAX_SIZE = 200
RANDOM_SEED = 42
EVAL_SAMPLE_SIZE: int | None = None  # None = 全部校验集
SEARCH_EVAL_SPLIT = "calib"

PROGRESS_EVERY = 10
# ==================================================


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in ("softmax", "ln1", "gelu", "ln2")
    ]


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "gelu":
        return tuple(
            level for level in POLY_LEVELS if gelu_level_allowed(layer_idx, level)
        )
    return POLY_LEVELS


def build_allowed_levels_table() -> list[tuple[int, ...]]:
    return [allowed_levels(layer_idx, kind) for layer_idx, kind in iter_positions()]


def is_scheme_legal(scheme: list[int], allowed_table: list[tuple[int, ...]]) -> bool:
    if len(scheme) != SCHEME_LEN:
        return False
    for idx, level in enumerate(scheme):
        if level not in allowed_table[idx]:
            return False
    return True


def random_legal_scheme(rng: random.Random, allowed_table: list[tuple[int, ...]]) -> list[int]:
    return [rng.choice(allowed_table[idx]) for idx in range(SCHEME_LEN)]


def random_scheme_with_mutation_draw(
    rng: random.Random, allowed_table: list[tuple[int, ...]]
) -> list[int]:
    scheme = random_legal_scheme(rng, allowed_table)
    idx = rng.randrange(SCHEME_LEN)
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
        population.append([max(levels) for levels in allowed_table])
    for _ in range(n_all_low):
        population.append([min(levels) for levels in allowed_table])
    while len(population) < n:
        population.append(random_legal_scheme(rng, allowed_table))
    rng.shuffle(population)
    return population


def compute_f_cost(task_name: str, scheme: list[int], *, cost_mode: str) -> int:
    validate_cost_scheme(scheme, task_name)
    if cost_mode == "bts":
        # TODO: return int(optimize_bootstrap(task_name, scheme))
        pass
    depth = int(compute_scheme_cost(task_name, scheme))
    return depth_sum_to_f_cost(depth)


@dataclass
class Individual:
    scheme: list[int]
    f_output_kl: float = 0.0
    f_cost: int = 0
    total_depth: int = 0
    accuracy_delta: float = 0.0
    loss_delta: float = 0.0
    f1_delta: float | None = None
    output_kl: float = 0.0
    flips_pct: float = 0.0
    rank: int = field(default=0, compare=False)
    crowding: float = field(default=0.0, compare=False)


def kl_not_worse(k1: float, k2: float) -> bool:
    return k1 <= k2


def kl_strictly_better(k1: float, k2: float) -> bool:
    return k1 < k2


def cost_strictly_better(c1: int, c2: int, *, cost_mode: str) -> bool:
    del cost_mode
    return c1 < c2


def cost_not_worse(c1: int, c2: int, *, cost_mode: str) -> bool:
    del cost_mode
    return c1 <= c2


def dominates(a: Individual, b: Individual, *, cost_mode: str) -> bool:
    if not kl_not_worse(a.f_output_kl, b.f_output_kl):
        return False
    if not cost_not_worse(a.f_cost, b.f_cost, cost_mode=cost_mode):
        return False
    kl_better = kl_strictly_better(a.f_output_kl, b.f_output_kl)
    cost_better = cost_strictly_better(a.f_cost, b.f_cost, cost_mode=cost_mode)
    return kl_better or cost_better


def evaluate_individual(
    ind: Individual,
    task_name: str,
    scheme_eval: SchemeEvaluator,
    *,
    cost_mode: str,
) -> Individual:
    metrics = scheme_eval.evaluate_for_scheme(ind.scheme)
    ind.f_output_kl = metrics.output_kl
    ind.f_cost = compute_f_cost(task_name, ind.scheme, cost_mode=cost_mode)
    ind.total_depth = int(compute_scheme_cost(task_name, ind.scheme))
    ind.accuracy_delta = metrics.accuracy_delta
    ind.loss_delta = metrics.loss_delta
    ind.f1_delta = metrics.f1_delta
    ind.output_kl = metrics.output_kl
    # NaN/Inf 不能参与支配：与任意有限 KL 均不可比，否则会混进 archive
    if not math.isfinite(ind.f_output_kl):
        ind.f_output_kl = math.inf
        ind.output_kl = math.inf
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
        ("f_output_kl", lambda x: x.f_output_kl),
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


def crossover(rng: random.Random, parent1: list[int], parent2: list[int]) -> list[int]:
    if rng.random() >= CROSSOVER_RATE:
        return parent1.copy()
    return [
        parent1[i] if rng.random() < 0.5 else parent2[i]
        for i in range(SCHEME_LEN)
    ]


def mutate(
    rng: random.Random,
    scheme: list[int],
    allowed_table: list[tuple[int, ...]],
) -> list[int]:
    child = scheme.copy()
    if rng.random() >= MUTATION_RATE:
        return child
    idx = rng.randrange(SCHEME_LEN)
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
            front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_output_kl))
            need = target_size - len(next_pop)
            next_pop.extend(front[:need])
            break
    return next_pop


class Archive:
    def __init__(self, *, cost_mode: str, max_size: int = ARCHIVE_MAX_SIZE) -> None:
        self.cost_mode = cost_mode
        self.max_size = max_size
        self.items: list[Individual] = []

    def consider(self, ind: Individual) -> None:
        # 非有限 KL：无法与有限目标比较，禁止进入 archive
        if not math.isfinite(ind.f_output_kl):
            return
        key = tuple(ind.scheme)
        if any(tuple(x.scheme) == key for x in self.items):
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
                f_output_kl=ind.f_output_kl,
                f_cost=ind.f_cost,
                total_depth=ind.total_depth,
                accuracy_delta=ind.accuracy_delta,
                loss_delta=ind.loss_delta,
                output_kl=ind.output_kl,
                f1_delta=ind.f1_delta,
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
                front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_output_kl))
                kept.extend(front[: self.max_size - len(kept)])
                break
        self.items = kept

    def sorted_items(self) -> list[Individual]:
        return sorted(self.items, key=lambda x: (x.f_cost, x.f_output_kl))


def reference_scheme_extremes(allowed_table: list[tuple[int, ...]]) -> tuple[list[int], list[int]]:
    all_low = [min(levels) for levels in allowed_table]
    all_high = [max(levels) for levels in allowed_table]
    return all_low, all_high


def print_eval_sanity(
    task_name: str,
    scheme_eval: SchemeEvaluator,
    allowed_table: list[tuple[int, ...]],
) -> None:
    all_low, all_high = reference_scheme_extremes(allowed_table)
    kl_base = scheme_eval.baseline_output_kl
    lo = scheme_eval.evaluate_for_scheme(all_low)
    hi = scheme_eval.evaluate_for_scheme(all_high)
    print(
        f"  校准（同一 eval 集）：baseline_KL={kl_base:.6f}  "
        f"all-low_KL={lo.output_kl:.6f}  "
        f"all-high_KL={hi.output_kl:.6f}  "
        f"baseline_acc={scheme_eval.baseline_acc:.4f}"
    )
    if scheme_eval.baseline_acc < 0.5 or lo.accuracy < 0.45:
        print(
            "  警告：准确率异常偏低，请检查 token_type_ids / 权重路径 / eval 子集"
        )


def run_evolution(
    task_name: str,
    scheme_eval: SchemeEvaluator,
    *,
    population_size: int = POPULATION_SIZE,
    num_generations: int = NUM_GENERATIONS,
    cost_mode: str = "depth",
    seed: int = RANDOM_SEED,
) -> tuple[list[Individual], Archive]:
    rng = random.Random(seed)
    allowed_table = build_allowed_levels_table()

    population = [
        Individual(scheme=scheme)
        for scheme in init_population_strategy(rng, allowed_table, population_size)
    ]
    print(f"  初始种群评估（{population_size} 个体 × 推理）...")
    for i, ind in enumerate(population, start=1):
        evaluate_individual(ind, task_name, scheme_eval, cost_mode=cost_mode)
        if i % max(1, population_size // 5) == 0 or i == population_size:
            print(f"    已评估 {i}/{population_size}")

    archive = Archive(cost_mode=cost_mode)
    archive.extend(population)

    for gen in range(1, num_generations + 1):
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
            evaluate_individual(child, task_name, scheme_eval, cost_mode=cost_mode)
            offspring.append(child)

        population = environmental_selection(
            population + offspring,
            population_size,
            cost_mode=cost_mode,
        )
        archive.extend(population)

        if gen % PROGRESS_EVERY == 0 or gen == num_generations:
            best_kl = min(ind.f_output_kl for ind in population)
            best = min(population, key=lambda x: (x.f_cost, x.f_output_kl))
            print(
                f"  代 {gen}/{num_generations}  "
                f"pop_best_KL={best_kl:.6f}  "
                f"archive={len(archive.items)}  "
                f"cache={len(scheme_eval._cache)}"
            )

    return archive.sorted_items(), archive


def format_scheme(scheme: list[int]) -> str:
    return str(scheme)


def save_pareto_csv(task_name: str, archive: Archive, path: str) -> None:
    """
    写出 Pareto CSV。非有限 KL 在搜索期已禁止入 archive，故不再写
    oor_count / oor_pct / n_eval_used；也不写 task、rank_hint。
    f_output_kl 与汇报后的 output_kl 一致，只保留 output_kl。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mrpc = task_name == "mrpc"
    header = [
        "f_cost",
        "total_depth",
        "accuracy_delta",
        "loss_delta",
        "output_kl",
        "flips_pct",
    ]
    if mrpc:
        header.append("f1_delta")
    header.append("scheme")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ind in archive.sorted_items():
            row = [
                ind.f_cost,
                ind.total_depth,
                fmt_metric_delta(ind.accuracy_delta),
                f"{ind.loss_delta:.8f}",
                f"{ind.output_kl:.8f}",
                fmt_flips_pct(ind.flips_pct),
            ]
            if mrpc:
                row.append(
                    fmt_metric_delta(ind.f1_delta)
                    if ind.f1_delta is not None
                    else ""
                )
            row.append(format_scheme(ind.scheme))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多目标进化（Output KL vs C_bts；搜索用校验集）"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE)
    parser.add_argument("--gens", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=EVAL_SAMPLE_SIZE,
        help="进化搜索用校验集抽样条数；默认全部校验集",
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help="进化搜索使用全部校验集（默认已是；保留兼容）",
    )
    parser.add_argument(
        "--calib-seed",
        type=int,
        default=CALIB_SEED,
        help=(
            f"读取 {{task}}_calib_indices_seed{{seed}}.json "
            f"（默认 {CALIB_SEED}）"
        ),
    )
    parser.add_argument(
        "--calib-indices-dir",
        default=CALIB_INDICES_DIR,
        help="calib 索引 JSON 目录（coverage_metric 输出，只读）",
    )
    parser.add_argument(
        "--cost-mode",
        choices=("depth", "bts"),
        default="depth",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    eval_samples: int | None = None if args.eval_full else args.eval_samples
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"多目标进化（Output KL 越小越好；cost={depth_f_cost_label()} 严格越小越好）"
    )
    print(
        f"device={device}, pop={args.pop}, gens={args.gens}, seed={args.seed}, "
        f"search_split={SEARCH_EVAL_SPLIT}, "
        f"eval={'full' if eval_samples is None else eval_samples}, "
        f"cost_mode={args.cost_mode}"
    )
    print(
        f"  calib indices: seed={args.calib_seed} "
        f"dir={args.calib_indices_dir}（只读，不重建）"
    )

    task_timings: list[tuple[str, float]] = []
    eval_common = dict(
        seed=args.seed,
        calib_seed=args.calib_seed,
        calib_indices_dir=args.calib_indices_dir,
    )

    for task_name in args.tasks:
        print(f"\n>>> {task_name.upper()}：加载模型与校验集...")
        try:
            scheme_eval = SchemeEvaluator(
                task_name,
                eval_samples=eval_samples,
                eval_split=SEARCH_EVAL_SPLIT,
                **eval_common,
            )
        except FileNotFoundError as exc:
            print(f"{task_name.upper()} 跳过：{exc}")
            continue
        print(f"  搜索评估集：{scheme_eval.eval_desc}")
        allowed_table = build_allowed_levels_table()
        print_eval_sanity(task_name, scheme_eval, allowed_table)

        t0 = time.perf_counter()
        _items, archive = run_evolution(
            task_name,
            scheme_eval,
            population_size=args.pop,
            num_generations=args.gens,
            cost_mode=args.cost_mode,
            seed=args.seed,
        )
        search_elapsed_s = time.perf_counter() - t0
        task_timings.append((task_name, search_elapsed_s))
        print(
            f"  搜索用时：{format_elapsed(search_elapsed_s)} ({search_elapsed_s:.2f}s)"
        )

        for report_split, suffix in (
            ("calib", "calib"),
            ("validation", "validation"),
        ):
            print(f"\n>>> {task_name.upper()}：Pareto {report_split} 汇报...")
            try:
                report_eval = SchemeEvaluator(
                    task_name,
                    eval_samples=None,
                    eval_split=report_split,
                    **eval_common,
                )
            except FileNotFoundError as exc:
                print(f"  跳过 {report_split} 汇报：{exc}")
                continue
            enrich_pareto_report(task_name, archive, report_eval)
            out_path = os.path.join(
                args.output_dir, f"{task_name}_pareto_kl_{suffix}.csv"
            )
            save_pareto_csv(task_name, archive, out_path)
            print(f"已保存：{out_path}")

    if task_timings:
        timings_path = save_search_timings(args.output_dir, task_timings)
        print(f"\n搜索用时已保存：{timings_path}")


if __name__ == "__main__":
    main()
