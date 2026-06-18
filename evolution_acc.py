"""
多目标进化搜索 POLY_SCHEMES 的 Pareto 前沿（|acc−原始acc| vs C_bts）。

目标：
  f_acc_delta = |验证集准确率 − 原始模型准确率|（越小越好；严格比较，无容差）
  f_cost      = ceil(深度和 / 10)（越小越好；严格比较，无容差）

评估集：任务开始时随机抽取 eval_samples 条（默认 200，固定索引），
        eval_samples=None 时使用全部验证集。

需 GPU + 已微调权重 + glue_datasets；每个体评估一次推理，耗时远大于 evolution_score。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from cost import NUM_LAYERS, compute_scheme_cost, validate_scheme as validate_cost_scheme
from gelu_poly import gelu_level_allowed
from poly_model_inference import (
    FINETUNED_MODEL_ROOT,
    LOCAL_DATA_ROOT,
    MAX_SEQ_LENGTH,
    SCHEME_ORIGINAL,
    apply_polynomial_scheme,
    get_preprocess_fn,
    install_eager_attention_poly_patch,
    is_original_scheme,
    scheme_index,
    validate_scheme,
    _load_tokenizer_and_model,
)

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]
POLY_LEVELS = (0, 1, 2)
OUTPUT_DIR = "./results/evolution_acc_results/"

POPULATION_SIZE = 80
NUM_GENERATIONS = 200
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / (NUM_LAYERS * 2)
TOURNAMENT_SIZE = 2
MAX_REGEN_ATTEMPTS = 32
ARCHIVE_MAX_SIZE = 80
RANDOM_SEED = 42
EVAL_SAMPLE_SIZE = 200

PROGRESS_EVERY = 10
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _QuietTrainer(Trainer):
    """屏蔽 Trainer.evaluate 每次打印的 metrics 字典。"""

    def log(self, logs, start_time=None) -> None:
        return


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in ("softmax", "gelu")
    ]


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "softmax":
        return POLY_LEVELS
    return tuple(
        level for level in POLY_LEVELS if gelu_level_allowed(layer_idx, level)
    )


def build_allowed_levels_table() -> list[tuple[int, ...]]:
    return [allowed_levels(layer_idx, kind) for layer_idx, kind in iter_positions()]


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
    return math.ceil(depth / 10)


class AccuracyEvaluator:
    """固定验证子集 + 单模型；scheme→accuracy 带缓存。"""

    def __init__(
        self,
        task_name: str,
        *,
        eval_samples: int | None = EVAL_SAMPLE_SIZE,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.task_name = task_name
        self.eval_samples = eval_samples
        self._cache: dict[tuple[int, ...], float] = {}
        self._metric = evaluate.load("accuracy")

        finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"模型不存在：{finetuned_model_path}")

        install_eager_attention_poly_patch()
        self.tokenizer, self.model, _ = _load_tokenizer_and_model(task_name)

        data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
        dataset = load_from_disk(data_path)
        val = dataset["validation"]
        n_total = len(val)

        if eval_samples is not None and eval_samples < n_total:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(n_total), eval_samples))
            val = val.select(indices)
            self.eval_desc = f"{eval_samples}/{n_total} 条（seed={seed} 固定）"
        else:
            self.eval_desc = f"全部 {n_total} 条"

        # 与 poly_model_inference.evaluate_on_validation 一致：不 set_format，
        # 保留 token_type_ids 等列，由 DataCollatorWithPadding 组 batch。
        self.eval_dataset = val.map(
            get_preprocess_fn(task_name, self.tokenizer),
            batched=True,
            desc=None,
        )
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return self._metric.compute(predictions=predictions, references=labels)

        self.trainer = _QuietTrainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=os.path.join(
                    OUTPUT_DIR, "_trainer_cache", task_name
                ),
                per_device_eval_batch_size=8,
                disable_tqdm=True,
                logging_strategy="no",
                log_level="error",
                report_to="none",
            ),
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        all_orig = [SCHEME_ORIGINAL] * (NUM_LAYERS * 2)
        self.baseline_acc = self.accuracy_for_scheme(all_orig)

    def _warmup_poly_kernels(self, scheme: list[int]) -> None:
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            gelu_warmup = torch.zeros(1, device=model_device, dtype=model_dtype)
            attn_warmup = torch.zeros(
                1, 1, 1, MAX_SEQ_LENGTH, device=model_device, dtype=model_dtype
            )
            key_valid_warmup = torch.ones_like(attn_warmup)
            for layer_idx in range(NUM_LAYERS):
                if not is_original_scheme(scheme[scheme_index(layer_idx, "gelu")]):
                    gelu_module = self.model.bert.encoder.layer[
                        layer_idx
                    ].intermediate.intermediate_act_fn
                    gelu_module.forward(gelu_warmup)
                if not is_original_scheme(scheme[scheme_index(layer_idx, "softmax")]):
                    attn_self = self.model.bert.encoder.layer[layer_idx].attention.self
                    poly_fn = getattr(attn_self, "_poly_softmax_fn", None)
                    if poly_fn is not None:
                        poly_fn(attn_warmup, dim=-1, key_valid_mask=key_valid_warmup)

    def accuracy_for_scheme(self, scheme: list[int]) -> float:
        key = tuple(scheme)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        validate_scheme(scheme, self.task_name)
        apply_polynomial_scheme(self.model, scheme, self.task_name)
        self._warmup_poly_kernels(scheme)
        self.model.eval()
        results = self.trainer.evaluate(self.eval_dataset)
        acc = float(results["eval_accuracy"])
        self._cache[key] = acc
        return acc


@dataclass
class Individual:
    scheme: list[int]
    f_acc_delta: float = 0.0
    f_cost: int = 0
    rank: int = field(default=0, compare=False)
    crowding: float = field(default=0.0, compare=False)


def acc_delta_not_worse(d1: float, d2: float) -> bool:
    return d1 <= d2


def acc_delta_strictly_better(d1: float, d2: float) -> bool:
    return d1 < d2


def cost_strictly_better(c1: int, c2: int, *, cost_mode: str) -> bool:
    del cost_mode
    return c1 < c2


def cost_not_worse(c1: int, c2: int, *, cost_mode: str) -> bool:
    del cost_mode
    return c1 <= c2


def dominates(a: Individual, b: Individual, *, cost_mode: str) -> bool:
    if not acc_delta_not_worse(a.f_acc_delta, b.f_acc_delta):
        return False
    if not cost_not_worse(a.f_cost, b.f_cost, cost_mode=cost_mode):
        return False
    acc_better = acc_delta_strictly_better(a.f_acc_delta, b.f_acc_delta)
    cost_better = cost_strictly_better(a.f_cost, b.f_cost, cost_mode=cost_mode)
    return acc_better or cost_better


def evaluate_individual(
    ind: Individual,
    task_name: str,
    acc_eval: AccuracyEvaluator,
    *,
    cost_mode: str,
) -> Individual:
    acc = acc_eval.accuracy_for_scheme(ind.scheme)
    ind.f_acc_delta = abs(acc - acc_eval.baseline_acc)
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
        ("f_acc_delta", lambda x: x.f_acc_delta),
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
            front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_acc_delta))
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
                f_acc_delta=ind.f_acc_delta,
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
                front.sort(key=lambda x: (-x.crowding, x.f_cost, x.f_acc_delta))
                kept.extend(front[: self.max_size - len(kept)])
                break
        self.items = kept

    def sorted_items(self) -> list[Individual]:
        return sorted(self.items, key=lambda x: (x.f_cost, x.f_acc_delta))


def reference_scheme_extremes(allowed_table: list[tuple[int, ...]]) -> tuple[list[int], list[int]]:
    all_low = [min(levels) for levels in allowed_table]
    all_high = [max(levels) for levels in allowed_table]
    return all_low, all_high


def print_eval_sanity(
    task_name: str,
    acc_eval: AccuracyEvaluator,
    allowed_table: list[tuple[int, ...]],
) -> None:
    all_low, all_high = reference_scheme_extremes(allowed_table)
    acc_base = acc_eval.baseline_acc
    acc_lo = acc_eval.accuracy_for_scheme(all_low)
    acc_hi = acc_eval.accuracy_for_scheme(all_high)
    print(
        f"  校准（同一 eval 集）：baseline={acc_base:.4f}  "
        f"all-low={acc_lo:.4f} (Δ={abs(acc_lo - acc_base):.4f})  "
        f"all-high={acc_hi:.4f} (Δ={abs(acc_hi - acc_base):.4f})"
    )
    if acc_base < 0.5 or acc_lo < 0.45:
        print(
            "  警告：准确率异常偏低，请检查 token_type_ids / 权重路径 / eval 子集"
        )


def run_evolution(
    task_name: str,
    acc_eval: AccuracyEvaluator,
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
        evaluate_individual(ind, task_name, acc_eval, cost_mode=cost_mode)
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
            evaluate_individual(child, task_name, acc_eval, cost_mode=cost_mode)
            offspring.append(child)

        population = environmental_selection(
            population + offspring,
            population_size,
            cost_mode=cost_mode,
        )
        archive.extend(population)

        if gen % PROGRESS_EVERY == 0 or gen == num_generations:
            best_delta = min(ind.f_acc_delta for ind in population)
            best = min(population, key=lambda x: (x.f_cost, x.f_acc_delta))
            print(
                f"  代 {gen}/{num_generations}  "
                f"pop_best_Δacc={best_delta:.6f}  "
                f"archive={len(archive.items)}  "
                f"cache={len(acc_eval._cache)}"
            )

    return archive.sorted_items(), archive


def format_scheme(scheme: list[int]) -> str:
    return str(scheme)


def save_pareto_csv(task_name: str, archive: Archive, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rank_hint", "f_acc_delta", "f_cost", "scheme"])
        for i, ind in enumerate(archive.sorted_items(), start=1):
            writer.writerow(
                [
                    task_name,
                    i,
                    f"{ind.f_acc_delta:.8f}",
                    ind.f_cost,
                    format_scheme(ind.scheme),
                ]
            )


def format_elapsed(seconds: float) -> str:
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
    eval_desc: str,
    elapsed_s: float | None = None,
) -> None:
    cost_label = "bts" if cost_mode == "bts" else "ceil(depth/10)"
    print(f"\n{'=' * 72}")
    print(
        f"Pareto 前沿：{task_name.upper()}  |  eval={eval_desc}  |  "
        f"cost={cost_label}  |  共 {len(items)} 个解"
    )
    if elapsed_s is not None:
        print(f"搜索用时：{format_elapsed(elapsed_s)} ({elapsed_s:.2f}s)")
    print(f"{'=' * 72}")
    print(f"{'#':<4} {'f_acc_Δ':<10} {'f_cost':<8} scheme")
    print("-" * 72)
    for i, ind in enumerate(items, start=1):
        print(
            f"{i:<4} {ind.f_acc_delta:<10.6f} {ind.f_cost:<8} "
            f"{format_scheme(ind.scheme)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多目标进化（|acc−原始acc| vs C_bts）"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE)
    parser.add_argument("--gens", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=EVAL_SAMPLE_SIZE,
        help="验证集抽样条数（默认 200）；与 --eval-full 互斥",
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help="使用全部验证集（忽略 --eval-samples）",
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

    print("多目标进化（|acc−原始acc| 越小越好；cost=ceil(depth/10) 严格越小越好）")
    print(
        f"device={device}, pop={args.pop}, gens={args.gens}, seed={args.seed}, "
        f"eval={'full' if eval_samples is None else eval_samples}, "
        f"cost_mode={args.cost_mode}"
    )

    task_timings: list[tuple[str, float]] = []

    for task_name in args.tasks:
        print(f"\n>>> {task_name.upper()}：加载模型与评估集...")
        try:
            acc_eval = AccuracyEvaluator(
                task_name, eval_samples=eval_samples, seed=args.seed
            )
        except FileNotFoundError as exc:
            print(f"{task_name.upper()} 跳过：{exc}")
            continue
        print(f"  评估集：{acc_eval.eval_desc}")
        allowed_table = build_allowed_levels_table()
        print_eval_sanity(task_name, acc_eval, allowed_table)

        t0 = time.perf_counter()
        items, archive = run_evolution(
            task_name,
            acc_eval,
            population_size=args.pop,
            num_generations=args.gens,
            cost_mode=args.cost_mode,
            seed=args.seed,
        )
        elapsed_s = time.perf_counter() - t0

        print_archive(
            task_name,
            items,
            cost_mode=args.cost_mode,
            eval_desc=acc_eval.eval_desc,
            elapsed_s=elapsed_s,
        )
        out_path = os.path.join(args.output_dir, f"{task_name}_pareto_acc.csv")
        save_pareto_csv(task_name, archive, out_path)
        print(f"已保存：{out_path}")
        task_timings.append((task_name, elapsed_s))

    if task_timings:
        print(f"\n{'=' * 72}")
        print("搜索用时汇总")
        print(f"{'=' * 72}")
        print(f"{'任务':<8} {'用时':<16} {'秒':<10}")
        print("-" * 36)
        total = 0.0
        for task_name, elapsed_s in task_timings:
            print(f"{task_name:<8} {format_elapsed(elapsed_s):<16} {elapsed_s:<10.2f}")
            total += elapsed_s
        print("-" * 36)
        print(f"{'合计':<8} {format_elapsed(total):<16} {total:<10.2f}")


if __name__ == "__main__":
    main()
