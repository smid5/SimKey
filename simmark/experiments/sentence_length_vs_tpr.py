import os
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False

from .utils import load_llm_config, test_watermark, load_prompts, COLORS, LINESTYLES, KEYS, METHODS, moving_average
from collections import defaultdict
from .tpr import compute_tpr

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_sentence_length_tpr(sentence_lengths, tprs, filename, fpr):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6, 4))  # wider plot

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
        plt.plot(
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

    plt.yscale("linear")  # Set y-axis to log scale to better capture small variations
    plt.xscale("linear")

    # Labels and ticks
    plt.xlabel("Sentence Length", fontsize=12)
    plt.ylabel(f"TPR @ FPR={int(fpr*100)}%", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # negative y moves legend below axes
        ncol=3
    )
    # plt.title(f"TPR at {int(fpr*100)}% FPR by Sentence Length", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def sentence_length_tpr(length_variations, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], fpr=1e-2, k=4, b=4, unwm_seeds=[42], wm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B', plot=True):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    tprs = defaultdict(dict)

    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue

        for method_name in method_names:
            for key_name in key_names:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}"
                print(f"Evaluating {method} at length {length}")

                pvals_unwm = [test_watermark(
                    applicable_prompts, length, llm_config, "nomark", method, seed=seed
                    ) for seed in unwm_seeds]
                pvals_wm = [test_watermark(
                    applicable_prompts, length, llm_config, method, method, seed=seed
                ) for seed in wm_seeds]
                tpr, _ = compute_tpr(pvals_unwm, pvals_wm, fpr)
                tprs[(method_name, key_name)][length] = tpr
        
    sorted_lengths = sorted(tprs[(method_names[0], key_names[0])].keys())
    for key in tprs:
        tprs[key] = [tprs[key][l] for l in sorted_lengths]
    if plot:
        plot_sentence_length_tpr(sorted_lengths, tprs, f"Figures/sentence_length_vs_tpr.pdf", fpr)
    else:
        return sorted_lengths, tprs, fpr