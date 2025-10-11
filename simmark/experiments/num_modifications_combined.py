import numpy as np
import matplotlib.pyplot as plt
from .num_modifications_vs_p_value import generate_p_value_modification_experiment
from .num_modifications_vs_tpr import generate_tpr_modification_experiment
from .utils import COLORS, LINESTYLES, moving_average

import matplotlib.pyplot as plt

def plot_pvalue(ax, modifications, p_values, xlabel):
    for (label, values) in p_values.items():
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
            modifications,
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

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r"Median $p$-value", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_tpr_fpr(ax, modifications, tprs, fpr, xlabel):
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
            modifications,
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
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(f"TPR @ FPR={int(fpr*100)}%", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(fontsize=11, frameon=False, loc="upper right")


def plot_num_modifications_combined(modification_values, num_tokens, filename, attack_name, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], fpr= 1e-3, k=4, b=4, wm_seeds=[42], unwm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(7, 4))

    modifications, tprs, fpr = generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names, key_names, fpr, k, b, wm_seeds, unwm_seeds, model_name, output_log_file=False, plot=False)
    _, p_values = generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names, key_names, k, b, wm_seeds, model_name, plot=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # (rows, cols)

    if attack_name=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack_name=="modify":
        xlabel="Number of Unrelated Token Replacements"
    elif attack_name=="translate":
        xlabel="Number of Translated Token Replacements"
    elif attack_name=="mask":
        xlabel="Number of Masked Token Replacements"
    else:
        xlabel="Number of Word Modifications"

    plot_tpr_fpr(axes[0], modifications, tprs, fpr, xlabel)
    plot_pvalue(axes[1], modifications, p_values, xlabel)

    # Optional: shared legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=16,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),  # negative y moves legend below axes
        ncol=3
    )

    plt.tight_layout()
    save_filename = f"Figures/num_modifications_{attack_name}_combined.pdf"
    plt.savefig(save_filename, bbox_inches="tight", dpi=300)
    plt.close()
