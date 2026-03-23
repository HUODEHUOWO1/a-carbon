from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_CSV = Path("runs/figures/quality_budget_sensitivity.csv")
OUT_DIR = Path("runs/figures")


def _prepare() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    g = (
        df.groupby(["workload", "quality_budget_pct"], as_index=False)
        .agg(
            mean_reduction=("carbon_reduction_pct", "mean"),
            std_reduction=("carbon_reduction_pct", "std"),
            n=("carbon_reduction_pct", "count"),
        )
        .sort_values(["workload", "quality_budget_pct"])
    )
    g["ci95"] = 1.96 * g["std_reduction"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1))
    return g


def redraw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = _prepare()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    palette = {"nlp": "#1b4f72", "vision": "#922b21"}
    labels = {"nlp": "NLP", "vision": "Vision"}

    for w in ["nlp", "vision"]:
        d = plot_df[plot_df["workload"] == w].sort_values("quality_budget_pct")
        x = d["quality_budget_pct"].to_numpy(float)
        y = d["mean_reduction"].to_numpy(float)
        e = d["ci95"].to_numpy(float)

        ax.plot(
            x,
            y,
            color=palette[w],
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=labels[w],
            zorder=3,
        )
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt="none",
            ecolor=palette[w],
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.9,
            zorder=2,
        )
        ax.fill_between(x, y - e, y + e, color=palette[w], alpha=0.10, zorder=1)

    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.2, zorder=0)
    ax.axhline(30.0, color="#888888", linestyle=":", linewidth=1.0, zorder=0)
    ylim_top = max(60.0, float(np.nanmax(plot_df["mean_reduction"] + plot_df["ci95"])) * 1.15)
    ax.set_ylim(0.0, ylim_top)
    ax.text(
        1.02,
        ylim_top * 0.95,
        "Predefined budget = 1.0%",
        fontsize=9,
        color="#444444",
        fontweight="bold",
    )
    ax.text(0.03, 30.8, "30% reference", fontsize=9, color="#666666", fontweight="bold")

    ax.set_xlabel("Allowable Quality Degradation Budget (%)")
    ax.set_ylabel("Carbon Reduction vs Static-HQ (%)")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlim(-0.05, 2.55)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.20)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    leg = ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
    leg.get_frame().set_alpha(0.9)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    png = OUT_DIR / "figure7_quality_budget_sensitivity_academic.png"
    pdf = OUT_DIR / "figure7_quality_budget_sensitivity_academic.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    fig.savefig(OUT_DIR / "figure7_quality_budget_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure7_quality_budget_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"saved_png={png}")
    print(f"saved_pdf={pdf}")


if __name__ == "__main__":
    redraw()
