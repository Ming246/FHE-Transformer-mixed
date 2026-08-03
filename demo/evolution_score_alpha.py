"""
demo 副本：evolution_score + 位置权重 α。

目标（均越小越好）：
  f_loss = Σ_j α_j · S_{j,k_j}   （无 α 时退化为 ΣS）
  f_cost = ceil(深度和 / COST_DEPTH_DIVISOR)

α 来自 demo/calibrate_alpha_s1.py 的 {task}_alpha.csv（按 pos_idx）。
输出默认写到 demo/results/evolution_alpha/（不改外部 results/）。

用法：
  python3 demo/evolution_score_alpha.py --tasks mrpc --pop 60 --gens 80 \\
    --sensitive-dir results/sensitive_scores_1 \\
    --alpha-csv demo/results/alpha_calib/mrpc/mrpc_alpha.csv \\
    --eval-samples 64
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)
from _repo import DEMO_RESULTS_DIR, REPO_ROOT, SENSITIVE_S1_DIR, chdir_repo

chdir_repo()

from cost import (
    NUM_LAYERS,
    SCHEME_LEN,
    compute_scheme_cost,
    depth_f_cost_label,
    depth_sum_to_f_cost,
    scheme_index,
    validate_scheme,
)
from gelu_poly import gelu_level_allowed
from evolution_infer import format_elapsed, save_search_timings

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
SENSITIVE_OUTPUT_DIR = SENSITIVE_S1_DIR
POLY_LEVELS = (0, 1, 2)
EVOLUTION_RESULTS_ROOT = os.path.join(DEMO_RESULTS_DIR, "evolution_alpha")


def sensitive_dir_tag(sensitive_dir: str) -> str:
    """从敏感度目录路径取末级名，用于区分不同 score 的进化结果。"""
    tag = os.path.basename(os.path.normpath(sensitive_dir))
    if not tag or tag in (".", ".."):
        raise ValueError(f"无法从敏感度目录解析标签：{sensitive_dir!r}")
    return tag


def evolution_output_dir(
    sensitive_dir: str = SENSITIVE_OUTPUT_DIR,
    *,
    root: str = EVOLUTION_RESULTS_ROOT,
) -> str:
    """results/evolution_results/<sensitive_scores_*>/"""
    return os.path.join(root, sensitive_dir_tag(sensitive_dir))


OUTPUT_DIR = evolution_output_dir(SENSITIVE_OUTPUT_DIR)

POPULATION_SIZE = 120
NUM_GENERATIONS = 300
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / SCHEME_LEN
TOURNAMENT_SIZE = 2
MAX_REGEN_ATTEMPTS = 32
ARCHIVE_MAX_SIZE = 80
RANDOM_SEED = 42

# loss 目标单一相对容差（10%）；由 τ 导出严格 / 不差阈值
LOSS_REL_TOL = 0.01
LOSS_STRICT_RATIO = 1.0 - LOSS_REL_TOL   # 0.9
LOSS_NOT_WORSE_RATIO = 1.0 + LOSS_REL_TOL  # 1.1

RUN_INFERENCE_ON_ARCHIVE = True
# ==================================================


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in ("softmax", "ln1", "gelu", "ln2")
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
    if len(out) != SCHEME_LEN:
        raise ValueError(f"{task_name} 敏感度行数应为 {SCHEME_LEN}，当前为 {len(out)}")
    return out


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "gelu":
        return tuple(
            level for level in POLY_LEVELS if gelu_level_allowed(layer_idx, level)
        )
    return POLY_LEVELS


def build_allowed_levels_table() -> list[tuple[int, ...]]:
    table: list[tuple[int, ...]] = []
    for layer_idx, kind in iter_positions():
        table.append(allowed_levels(layer_idx, kind))
    return table


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
    """随机 scheme，某位可能抽到非法档位（用于变异路径）。"""
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


def load_alpha_weights(alpha_csv: str | None) -> dict[tuple[int, str], float]:
    """位置权重；缺省全 1。CSV 需含 pos_idx / layer_idx / kind / alpha。"""
    weights = {(layer_idx, kind): 1.0 for layer_idx, kind in iter_positions()}
    if not alpha_csv:
        return weights
    if not os.path.isfile(alpha_csv):
        raise FileNotFoundError(f"alpha CSV 不存在：{alpha_csv}")
    with open(alpha_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer_idx = int(row["layer_idx"])
            kind = row["kind"]
            weights[(layer_idx, kind)] = float(row["alpha"])
    return weights


def compute_f_loss(
    scheme: list[int],
    s_mat: dict[tuple[int, str], dict[int, float]],
    alpha: dict[tuple[int, str], float] | None = None,
) -> float:
    total = 0.0
    for layer_idx, kind in iter_positions():
        level = scheme[scheme_index(layer_idx, kind)]
        w = 1.0 if alpha is None else float(alpha[(layer_idx, kind)])
        total += w * s_mat[(layer_idx, kind)][level]
    return total


def compute_f_cost(task_name: str, scheme: list[int], *, cost_mode: str) -> int:
    """
    f_cost = C_bts(scheme)。

    cost_mode='depth'：ceil(深度和 / COST_DEPTH_DIVISOR)。
    cost_mode='bts'  ：预留，当前仍回退 depth 映射，待 bts 求解器接入。
    """
    validate_scheme(scheme, task_name)
    if cost_mode == "bts":
        # TODO: return int(optimize_bootstrap(task_name, scheme))
        pass
    depth = int(compute_scheme_cost(task_name, scheme))
    return depth_sum_to_f_cost(depth)


@dataclass
class Individual:
    scheme: list[int]
    f_loss: float = 0.0
    f_cost: int = 0
    total_depth: int = 0
    accuracy_delta: float = 0.0
    loss_delta: float = 0.0
    output_kl: float = 0.0
    flips_pct: float = 0.0
    oor_count: int = 0
    oor_pct: float = 0.0
    n_eval_used: int = 0
    f1_delta: float | None = None
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
    del cost_mode
    return c1 < c2


def cost_not_worse(c1: int, c2: int, *, cost_mode: str) -> bool:
    del cost_mode
    return c1 <= c2


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
    alpha: dict[tuple[int, str], float] | None = None,
) -> Individual:
    ind.f_loss = compute_f_loss(ind.scheme, s_mat, alpha=alpha)
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
    alpha: dict[tuple[int, str], float] | None = None,
) -> tuple[list[Individual], Archive]:
    rng = random.Random(seed)
    s_mat = load_sensitivity_matrix(task_name, sensitive_dir)
    allowed_table = build_allowed_levels_table()

    population = [
        Individual(scheme=scheme)
        for scheme in init_population_strategy(rng, allowed_table, population_size)
    ]
    for ind in population:
        evaluate_individual(
            ind, task_name, s_mat, cost_mode=cost_mode, alpha=alpha
        )

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
            evaluate_individual(
                child, task_name, s_mat, cost_mode=cost_mode, alpha=alpha
            )
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


def save_pareto_csv(task_name: str, archive: Archive, path: str, *, with_inference: bool) -> None:
    from poly_model_inference import fmt_metric_delta

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mrpc = task_name == "mrpc"
    header = ["task", "rank_hint", "f_loss", "f_cost"]
    if with_inference:
        header.extend(
            [
                "total_depth",
                "accuracy_delta",
                "loss_delta",
                "output_kl",
                "flips_pct",
                "oor_count",
                "oor_pct",
                "n_eval_used",
            ]
        )
        if mrpc:
            header.append("f1_delta")
    header.append("scheme")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, ind in enumerate(archive.sorted_items(), start=1):
            row = [task_name, i, f"{ind.f_loss:.8e}", ind.f_cost]
            if with_inference:
                from evolution_infer import fmt_flips_pct, fmt_oor_pct

                row.extend(
                    [
                        ind.total_depth,
                        fmt_metric_delta(ind.accuracy_delta),
                        f"{ind.loss_delta:.8f}",
                        f"{ind.output_kl:.8f}",
                        fmt_flips_pct(ind.flips_pct),
                        ind.oor_count,
                        fmt_oor_pct(ind.oor_pct),
                        ind.n_eval_used,
                    ]
                )
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
        description="demo：ΣαS vs depth 的 NSGA-II（evolution_score 副本）"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mrpc"],
        help="任务列表（demo 默认仅 mrpc）",
    )
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE, help="种群规模")
    parser.add_argument("--gens", type=int, default=NUM_GENERATIONS, help="进化代数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    parser.add_argument(
        "--cost-mode",
        choices=("depth", "bts"),
        default="depth",
        help=f"depth={depth_f_cost_label()}; bts=预留",
    )
    parser.add_argument(
        "--sensitive-dir",
        default=SENSITIVE_OUTPUT_DIR,
        help="敏感度 CSV 目录（默认 sensitive_scores_1）",
    )
    parser.add_argument(
        "--alpha-csv",
        default=None,
        help="α CSV / 目录 / 含 {task} 的路径模板；缺省 α≡1",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"输出目录；默认 {EVOLUTION_RESULTS_ROOT}/",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Pareto 汇报样本数；缺省全量；>0 子集",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="跳过 Pareto 验证集推理",
    )
    args = parser.parse_args()

    with_inference = not args.skip_inference
    if args.output_dir is None:
        args.output_dir = EVOLUTION_RESULTS_ROOT
    os.makedirs(args.output_dir, exist_ok=True)

    print("多目标进化搜索（demo / ΣαS）")
    print(
        f"pop={args.pop}, gens={args.gens}, seed={args.seed}, "
        f"cost_mode={args.cost_mode}, sensitive_dir={args.sensitive_dir}"
    )
    print(f"alpha_csv={args.alpha_csv or '(none → α=1)'}")
    print(f"output_dir={args.output_dir}")
    print(f"REPO_ROOT={REPO_ROOT}")

    task_timings: list[tuple[str, float]] = []

    for task_name in args.tasks:
        alpha_path = args.alpha_csv
        if alpha_path and "{task}" in alpha_path:
            alpha_path = alpha_path.format(task=task_name)
        if alpha_path and os.path.isdir(alpha_path):
            # 支持 demo/results/alpha_calib/ 或 .../alpha_calib/<task>/
            cand = os.path.join(alpha_path, f"{task_name}_alpha.csv")
            cand_nested = os.path.join(alpha_path, task_name, f"{task_name}_alpha.csv")
            if os.path.isfile(cand):
                alpha_path = cand
            elif os.path.isfile(cand_nested):
                alpha_path = cand_nested
            else:
                alpha_path = cand_nested
        try:
            alpha = load_alpha_weights(alpha_path)
        except FileNotFoundError as exc:
            print(f"\n{task_name.upper()} 跳过：{exc}")
            continue

        try:
            t0 = time.perf_counter()
            _items, archive = run_evolution(
                task_name,
                population_size=args.pop,
                num_generations=args.gens,
                cost_mode=args.cost_mode,
                seed=args.seed,
                sensitive_dir=args.sensitive_dir,
                alpha=alpha,
            )
            search_elapsed_s = time.perf_counter() - t0
        except FileNotFoundError as exc:
            print(f"\n{task_name.upper()} 跳过：{exc}")
            continue

        task_timings.append((task_name, search_elapsed_s))
        print(
            f"  {task_name.upper()} 搜索用时："
            f"{format_elapsed(search_elapsed_s)} ({search_elapsed_s:.2f}s)"
        )

        if with_inference:
            from evolution_infer import SchemeEvaluator, enrich_pareto_report

            eval_samples = args.eval_samples
            if eval_samples is not None and eval_samples <= 0:
                eval_samples = None
            print(
                f"\n>>> {task_name.upper()}：Pareto 汇报"
                f"（samples={eval_samples if eval_samples else 'full'}）..."
            )
            try:
                report_eval = SchemeEvaluator(
                    task_name,
                    eval_samples=eval_samples,
                    seed=args.seed,
                )
            except FileNotFoundError as exc:
                print(f"  推理跳过：{exc}")
                with_inference_task = False
            else:
                enrich_pareto_report(task_name, archive, report_eval)
                with_inference_task = True
        else:
            with_inference_task = False

        out_path = os.path.join(args.output_dir, f"{task_name}_pareto.csv")
        save_pareto_csv(
            task_name, archive, out_path, with_inference=with_inference_task
        )
        print(f"已保存：{out_path}")

    if task_timings:
        timings_path = save_search_timings(args.output_dir, task_timings)
        print(f"\n搜索用时已保存：{timings_path}")


if __name__ == "__main__":
    main()
