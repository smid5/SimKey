import numpy as np
import matplotlib.pyplot as plt
from .sentence_length_vs_tpr import sentence_length_tpr
from .plot_distortion_dist import sentence_length_median_distortion
from .utils import COLORS, LINESTYLES, moving_average

import matplotlib.pyplot as plt

def plot_distortion(ax, sentence_lengths, distortions):
    for (label, values) in distortions.items():
        if isinstance(label, tuple):
            method_name, key_name = label
            color = COLORS[method_name]
            linestyle = LINESTYLES[key_name]
            legend_label = f"{method_name} ({key_name})"
        else:
            method_name = label
            color = COLORS[method_name]
            linestyle = "-"
            legend_label = method_name

        moving_avg = moving_average(values, window_size=3)
        ax.plot(
            sentence_lengths,
            moving_avg,
            marker="None",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle=linestyle,
            color=color,
            linewidth=2,
            label=legend_label,
            alpha=0.9
        )

    ax.set_xlabel("Sentence Length", fontsize=18)
    ax.set_ylabel("Median Perplexity", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_tpr_fpr(ax, sentence_lengths, tprs, fpr):
    for (label, values) in tprs.items():
        if isinstance(label, tuple):
            method_name, key_name = label
            color = COLORS[method_name]
            linestyle = LINESTYLES[key_name]
            legend_label = f"{method_name} ({key_name})"
        else:
            method_name = label
            color = COLORS[method_name]
            linestyle = "-"
            legend_label = method_name

        moving_avg = moving_average(values, window_size=3)
        ax.plot(
            sentence_lengths,
            moving_avg,
            marker="None",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle=linestyle,
            color=color,
            linewidth=2,
            label=legend_label,
            alpha=0.9
        )
    ax.set_xlabel("Sentence Length", fontsize=18)
    ax.set_ylabel(f"TPR @ FPR={int(fpr*100)}%", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(fontsize=11, frameon=False, loc="upper right")


def plot_sentence_length_combined(length_variations, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], fpr=1e-2, k=4, b=4, unwm_seeds=[42], wm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(7, 4))

    sorted_lengths, tprs, fpr = sentence_length_tpr(length_variations, filename, method_names, key_names, fpr, k, b, unwm_seeds, wm_seeds, model_name, plot=False)
    _, distortions = sentence_length_median_distortion(length_variations, filename, method_names, key_names, k, b, wm_seeds, model_name, plot=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # (rows, cols)

    plot_tpr_fpr(axes[0], sorted_lengths, tprs, fpr)
    plot_distortion(axes[1], sorted_lengths, distortions)

    # Optional: shared legend for both subplots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=16,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),  # negative y moves legend below axes
        ncol=4
    )

    plt.tight_layout()
    save_filename = "Figures/sentence_length_combined.pdf"
    plt.savefig(save_filename, bbox_inches="tight", dpi=300)
    plt.close()
