import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, METHODS, COLORS, KEYS, LINESTYLES, moving_average
from collections import defaultdict
from .tpr import compute_tpr


def plot_tpr_modifications(modifications, tprs, filename, xlabel, fpr):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6, 4))

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
            modifications,
            values,
            marker="None",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle=linestyle,
            color=color,
            linewidth=2,
            label=legend_label,
            alpha=0.9
        )

    # Labels and ticks
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(f"TPR @ FPR={int(fpr*100)}%", fontsize=14)
    plt.xticks(modifications, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(f"TPR at {int(fpr*100)}% FPR by Number of Translated Token Replacements", fontsize=18)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # negative y moves legend below axes
        ncol=3
    )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], fpr= 1e-3, k=4, b=4, wm_seeds=[42], unwm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B', output_log_file=False, plot=True):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    modifications = np.array(modification_values)

    tprs = defaultdict(list)
    pvals_unwm = defaultdict(list)

    for method_name in method_names:
        for key_name in key_names:
            method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
            pvals_unwm[(method_name, key_name)] = [test_watermark(
                prompts, num_tokens, llm_config, "nomark", method, seed=seed
            ) for seed in unwm_seeds]
            
    for num_modify in modification_values:
        for method_name in method_names:
            for key_name in key_names:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
                print(f"Evaluating {method} with {attack_name} attack and {num_modify} modifications")

                p_vals = [test_watermark(
                    prompts, num_tokens, llm_config, method, method, f"{attack_name}_{num_modify}", seed=seed
                ) for seed in wm_seeds]
                tpr, _ = compute_tpr(pvals_unwm[(method_name, key_name)], p_vals, fpr)
                tprs[(method_name, key_name)].append(tpr)

    save_filename = f"Figures/tpr_vs_{attack_name}_attack_k{k}_b{b}.pdf"
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
    if output_log_file:
        log_filename = f"logs/tpr_vs_attack_k{k}_b{b}.txt"
        if attack_name=="modify":
            attack = "unrelated token replacements"
        elif attack_name=="mask":
            attack = "related token replacements"
        with open(log_filename, "a") as f:
            for (method_name, key_name), tpr_values in tprs.items():
                for tpr_value, num_modify in zip(tpr_values, modification_values):
                    f.write(f"TPR for {method_name} ({key_name}) under {num_modify} {attack}: {tpr_value}\n")
            print("\n")
    if plot:
        #Generate plot
        plot_tpr_modifications(modifications, tprs, save_filename, xlabel, fpr)
    else:
        return modifications, tprs, fpr
    
    