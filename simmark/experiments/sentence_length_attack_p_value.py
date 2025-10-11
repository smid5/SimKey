import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_watermark, load_prompts, METHODS, COLORS, KEYS, LINESTYLES
from collections import defaultdict
import numpy as np

def plot_p_value_modifications(modifications, p_values, filename, xlabel):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(10, 4.5))  # wider aspect ratio

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

        plt.plot(
            modifications,
            values,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1,
            linestyle=linestyle,
            color=color,
            linewidth=2,
            label=legend_label
        )

    # Labels and ticks
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Median of median token-level p-values", fontsize=14)
    plt.xticks(modifications, fontsize=12)
    plt.yticks(fontsize=12)

    # Grid styling
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend outside
    plt.legend(fontsize=11, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names=["ExpMin", "SynthID"], key_names=["Standard Hashing", "SimHash", "No Hashing"], k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    modifications = np.array(modification_values)

    p_values = defaultdict(list)

    for i, num_modify in enumerate(modification_values):
        for method_name in method_names:
            for key_name in key_names:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"

                p_vals = [test_watermark(
                    prompts, num_tokens, llm_config, method, method, f"{attack_name}_{num_modify}", seed=seed
                ) for seed in seeds]
                median_pval = np.median(p_vals)
                p_values[(method_name, key_name)][i] = median_pval

    save_filename = f"figures/p_value_vs_{attack_name}_attack_k{k}_b{b}.pdf"
    if attack_name=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack_name=="modify":
        xlabel="Number of Unrelated Word Replacements"
    elif attack_name=="translate":
        xlabel="Number of Translated Word Replacements"
    elif attack_name=="mask":
        xlabel="Number of Masked Word Replacements"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    plot_p_value_modifications(modifications, p_values, save_filename, xlabel)