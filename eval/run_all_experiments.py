"""Run baseline, ablation, and testA diagnosis experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_ROOT = Path("outputs") / "experiments"
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def run_cmd(args: list[str]) -> None:
    print("$ " + " ".join(args))
    subprocess.run(args, check=True)


def safe_load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"Warning: missing result file: {path}")
    return None


def ensure_annotations(data_root: Path) -> Path:
    annotations = data_root / "annotations.json"
    readme = data_root / "README.md"
    if annotations.exists():
        return annotations
    if not readme.exists():
        raise FileNotFoundError(
            f"Missing {annotations} and cannot generate it because {readme} is absent."
        )
    run_cmd(
        [
            sys.executable,
            "generate_annotations.py",
            "--image-dir",
            str(data_root),
            "--readme",
            str(readme),
            "--output",
            str(annotations),
        ]
    )
    return annotations


def plot_baseline(baseline: dict, baseline_dir: Path) -> None:
    summary = baseline.get("summary", baseline)
    method_keys = [
        "center_crop_mean_iou",
        "rule_based_mean_iou",
        "saliency_only_mean_iou",
        "aesthetic_only_mean_iou",
        "yolo_only_mean_iou",
        "full_mean_iou",
    ]
    ious = [summary.get(k, 0.0) for k in method_keys]
    df_base = pd.DataFrame({"Method": method_keys, "Mean IoU": ious})
    df_base.to_csv(baseline_dir / "baseline_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(
        data=df_base,
        x="Method",
        y="Mean IoU",
        hue="Method",
        palette="Blues_d",
        legend=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean IoU")
    plt.title("Baseline Comparison")
    for bar, iou in zip(bars.patches, ious):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{iou:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_bar.png")
    plt.close()

    if "results" not in baseline or not baseline["results"]:
        return

    iou_matrix = []
    for row in baseline["results"]:
        iou_matrix.append(
            [
                row.get("center_crop_iou", 0),
                row.get("rule_based_iou", 0),
                row.get("saliency_only_iou", 0),
                row.get("aesthetic_only_iou", 0),
                row.get("yolo_only_iou", 0),
                row.get("full_iou", 0),
            ]
        )
    df_iou = pd.DataFrame(iou_matrix, columns=method_keys)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_iou.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("IoU Correlation between Methods")
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_correlation_heatmap.png")
    plt.close()

    df_melt = df_iou.melt(var_name="Method", value_name="IoU")
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df_melt,
        x="Method",
        y="IoU",
        hue="Method",
        palette="Set2",
        legend=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("IoU Distribution across Methods")
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_violin.png")
    plt.close()


def plot_ablation(ablation: dict, ablation_dir: Path) -> None:
    module_results = ablation.get("module_ablation", [])
    if module_results:
        df_module = pd.DataFrame(module_results)
        df_module = df_module[["name", "mean_iou"]].rename(
            columns={"name": "Module", "mean_iou": "mIoU"}
        )
        full_rows = df_module[df_module["Module"] == "full"]
        full_iou = float(full_rows["mIoU"].iloc[0]) if not full_rows.empty else 1.0
        df_module["Relative Performance"] = df_module["mIoU"] / max(full_iou, 1e-9) * 100
        df_module.to_csv(ablation_dir / "module_ablation.csv", index=False)

        plt.figure(figsize=(12, 6))
        bars = sns.barplot(
            data=df_module,
            x="Module",
            y="Relative Performance",
            hue="Module",
            palette="RdYlGn",
            legend=False,
        )
        plt.axhline(100, color="r", linestyle="--", label="Full model (100%)")
        plt.ylabel("Relative mIoU (%)")
        plt.title("Module Ablation")
        plt.xticks(rotation=45, ha="right")
        for bar, val in zip(bars.patches, df_module["Relative Performance"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )
        plt.legend()
        plt.tight_layout()
        plt.savefig(ablation_dir / "module_ablation_bar.png")
        plt.close()

    k_results = ablation.get("k_ablation", [])
    if not k_results:
        print("Warning: no k_ablation field found in ablation results.")
        return

    df_k = pd.DataFrame(k_results)
    df_k["K"] = df_k["name"].str.replace("K=", "", regex=False).astype(int)
    df_k = df_k.sort_values("K")
    df_k.to_csv(ablation_dir / "k_ablation.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_k["K"], df_k["mean_iou"], "b-o", label="mIoU")
    ax1.set_xlabel("Number of Candidates (K)")
    ax1.set_ylabel("Mean IoU", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(df_k["K"], df_k["mean_time"], "r-s", label="Time (s)")
    ax2.set_ylabel("Time per image (s)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    plt.title("Effect of Candidate Count K")
    fig.tight_layout()
    plt.savefig(ablation_dir / "k_ablation_line.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df_k["mean_time"],
        df_k["mean_iou"],
        c=df_k["K"],
        cmap="viridis",
        s=100,
        edgecolors="k",
    )
    plt.colorbar(scatter, label="K value")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean IoU")
    plt.title("Trade-off: IoU vs Inference Time")
    plt.tight_layout()
    plt.savefig(ablation_dir / "k_ablation_pareto.png")
    plt.close()

    df_k["iou_gain"] = df_k["mean_iou"].diff()
    df_k["time_gain"] = df_k["mean_time"].diff()
    df_k["efficiency"] = df_k["iou_gain"] / df_k["time_gain"]
    plt.figure(figsize=(8, 5))
    plt.plot(df_k["K"][1:], df_k["efficiency"][1:], "g-o")
    plt.xlabel("K")
    plt.ylabel("Marginal IoU Gain per Second")
    plt.title("Efficiency of Increasing K")
    plt.axhline(y=0, color="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(ablation_dir / "k_ablation_efficiency.png")
    plt.close()


def main() -> None:
    data_root = Path("testA/testA")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_dir = OUTPUT_ROOT / "baseline"
    ablation_dir = OUTPUT_ROOT / "ablation"
    testa_dir = OUTPUT_ROOT / "testa_diagnose"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    testa_dir.mkdir(parents=True, exist_ok=True)

    annotations = ensure_annotations(data_root)
    baseline_json = baseline_dir / "baseline_results.json"
    ablation_json = ablation_dir / "ablation_results.json"

    print("=== Running baseline comparison ===")
    run_cmd(
        [
            sys.executable,
            "eval/baseline_compare.py",
            "--annotations",
            str(annotations),
            "--image-root",
            str(data_root),
            "--output",
            str(baseline_json),
        ]
    )

    print("=== Running ablation experiments ===")
    run_cmd(
        [
            sys.executable,
            "eval/ablation.py",
            "--annotations",
            str(annotations),
            "--image-root",
            str(data_root),
            "--output",
            str(ablation_json),
        ]
    )

    print("=== Running testA diagnosis ===")
    run_cmd(
        [
            sys.executable,
            "eval/testa_diagnose.py",
            "--dataset-dir",
            str(data_root),
            "--output-dir",
            str(testa_dir),
        ]
    )

    baseline = safe_load_json(baseline_json)
    if baseline:
        plot_baseline(baseline, baseline_dir)

    ablation = safe_load_json(ablation_json)
    if ablation:
        plot_ablation(ablation, ablation_dir)

    print(f"\nAll experiment outputs saved to: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
