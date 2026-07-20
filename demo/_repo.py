"""demo 内脚本共用：定位仓库根目录，并把根加入 sys.path（只读外部代码）。"""
from __future__ import annotations

import os
import sys

DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 外部只读数据默认路径（相对仓库根）
SENSITIVE_S1_DIR = os.path.join(REPO_ROOT, "results", "sensitive_scores_1")
EVOLUTION_S1_DIR = os.path.join(
    REPO_ROOT, "results", "evolution_results", "sensitive_scores_1"
)
EVOLUTION_KL_DIR = os.path.join(REPO_ROOT, "results", "evolution_kl_results")

# demo 输出
DEMO_RESULTS_DIR = os.path.join(DEMO_DIR, "results")


def chdir_repo() -> None:
    """外部脚本用相对路径 ./glue_datasets / ./finetuned_weight，需在仓库根 cwd。"""
    os.chdir(REPO_ROOT)
