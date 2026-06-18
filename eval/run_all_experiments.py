import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

OUTPUT_ROOT = Path("三个实验outputs")
OUTPUT_ROOT.mkdir(exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


def safe_load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    else:
        print(f"警告: {path} 不存在")
        return None


def main():
    data_root = "testA/testA"
    baseline_dir = OUTPUT_ROOT / "baseline"
    ablation_dir = OUTPUT_ROOT / "ablation"
    testa_dir = OUTPUT_ROOT / "testa_diagnose"
    baseline_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)
    testa_dir.mkdir(exist_ok=True)

    baseline_json = baseline_dir / "baseline_results.json"
    ablation_json = ablation_dir / "ablation_results.json"

    print("=== 运行基线实验 ===")
    run_cmd(
        f"python eval/baseline_compare.py --annotations {data_root}/annotations.json --image-root {data_root} --output {baseline_json}"
    )
    print("=== 运行消融实验 ===")
    run_cmd(
        f"python eval/ablation.py --annotations {data_root}/annotations.json --image-root {data_root} --output {ablation_json}"
    )
    print("=== 运行 testA 诊断 ===")
    run_cmd(
        f"python eval/testa_diagnose.py --dataset-dir {data_root} --output {testa_dir}"
    )

    # ----------------------------- 基线可视化 -----------------------------
    baseline = safe_load_json(baseline_json)
    if baseline:
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

        # 柱状图
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

        # 如果有每个图像的 IoU，绘制小提琴图和热力图
        if "results" in baseline and len(baseline["results"]) > 0:
            iou_matrix = []
            for r in baseline["results"]:
                row = [
                    r.get("center_crop_iou", 0),
                    r.get("rule_based_iou", 0),
                    r.get("saliency_only_iou", 0),
                    r.get("aesthetic_only_iou", 0),
                    r.get("yolo_only_iou", 0),
                    r.get("full_iou", 0),
                ]
                iou_matrix.append(row)
            df_iou = pd.DataFrame(iou_matrix, columns=method_keys)
            corr = df_iou.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
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

    # ----------------------------- 消融可视化 -----------------------------
    ablation = safe_load_json(ablation_json)
    if ablation:
        module_results = ablation.get("module_ablation", [])
        if module_results:
            df_module = pd.DataFrame(module_results)
            df_module = df_module[["name", "mean_iou"]].rename(
                columns={"name": "Module", "mean_iou": "mIoU"}
            )
            full_iou = df_module[df_module["Module"] == "full"]["mIoU"].values[0]
            df_module["Relative Performance"] = df_module["mIoU"] / full_iou * 100
            df_module.to_csv(ablation_dir / "module_ablation.csv", index=False)

            # 相对性能柱状图
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
            plt.title("Module Ablation: Performance Drop after Removing Component")
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

            # 雷达图
            labels = df_module["Module"].tolist()
            drops = (100 - df_module["Relative Performance"]).tolist()
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            drops += drops[:1]
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
            ax.plot(angles, drops, "o-", linewidth=2)
            ax.fill(angles, drops, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylabel("IoU Drop (%)")
            ax.set_title("Contribution of Each Module (Higher Drop = More Important)")
            plt.tight_layout()
            plt.savefig(ablation_dir / "module_ablation_radar.png")
            plt.close()

        k_results = ablation.get("k_ablation", [])
        if k_results:
            df_k = pd.DataFrame(k_results)
            df_k["K"] = df_k["name"].str.replace("K=", "").astype(int)
            df_k = df_k.sort_values("K")
            df_k.to_csv(ablation_dir / "k_ablation.csv", index=False)

            # 双轴折线图
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

            # 帕累托前沿散点图
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                df_k["mean_time"],
                df_k["mean_iou"],
                c=df_k["K"],
                cmap="viridis",
                s=100,
                edgecolors="k",
            )
            plt.colorbar(sc, label="K value")
            plt.xlabel("Time (s)")
            plt.ylabel("Mean IoU")
            plt.title("Trade-off: IoU vs Inference Time (Pareto Front)")
            pareto = []
            for i, row in df_k.iterrows():
                dominated = False
                for j, row2 in df_k.iterrows():
                    if (
                        row2["mean_time"] <= row["mean_time"]
                        and row2["mean_iou"] >= row["mean_iou"]
                        and (
                            row2["mean_time"] < row["mean_time"]
                            or row2["mean_iou"] > row["mean_iou"]
                        )
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(row)
            pareto_df = pd.DataFrame(pareto)
            if not pareto_df.empty:
                plt.plot(
                    pareto_df["mean_time"],
                    pareto_df["mean_iou"],
                    "r--",
                    marker="D",
                    linewidth=2,
                    label="Pareto front",
                )
                for _, row in pareto_df.iterrows():
                    plt.annotate(
                        f"K={int(row['K'])}",
                        (row["mean_time"], row["mean_iou"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                    )
                plt.legend()
            plt.tight_layout()
            plt.savefig(ablation_dir / "k_ablation_pareto.png")
            plt.close()

            # 效率收益曲线
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
        else:
            print("⚠️ 未找到 K 值实验结果，请确保 ablation.py 正确输出 k_ablation 字段")

    print(f"\n✅ 所有结果已保存到: {OUTPUT_ROOT.absolute()}")
    print("  - 基线结果: baseline/ (含柱状图、相关性热力图、小提琴图)")
    print(
        "  - 消融结果: ablation/ (含相对性能柱状图、雷达图、K 值折线图、帕累托散点图、效率曲线)"
    )


if __name__ == "__main__":
    main()
