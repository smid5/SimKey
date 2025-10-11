import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, METHODS, COLORS, KEYS, LINESTYLES, moving_average
from collections import defaultdict


def plot_p_value_modifications(modifications, p_values, filename, xlabel):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6, 4))

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
        plt.plot(
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

    plt.yscale("log")
    # Labels and ticks
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Median p-value", fontsize=14)
    plt.xticks(modifications, fontsize=14)
    plt.yticks(fontsize=14)

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

def generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B', plot=True):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    modifications = np.array(modification_values)

    p_values = defaultdict(list)

    for num_modify in modification_values:
        for method_name in method_names:
            for key_name in key_names:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
                print(f"Evaluating {method} with {attack_name} attack and {num_modify} modifications")

                p_vals = [test_watermark(
                    prompts, num_tokens, llm_config, method, method, f"{attack_name}_{num_modify}", seed=seed
                ) for seed in seeds]
                median_pval = np.median(p_vals)
                p_values[(method_name, key_name)].append(median_pval)

    save_filename = f"Figures/p_value_vs_{attack_name}_attack_k{k}_b{b}.pdf"
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
    if plot:
        # Generate plot
        plot_p_value_modifications(modifications, p_values, save_filename, xlabel)
    else:
        return modifications, p_values

def subplot_p_value_modifications(modifications, p_values_dict, ax, xlabel):
    for idx, (label, values) in enumerate(p_values_dict.items()):
        ax.plot(modifications, values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)

    ax.set_yscale("linear")
    ax.set_xlabel(xlabel, fontsize=24)
    ax.legend()
    ax.grid()
    ax.set_xticks(modifications)
    ax.tick_params(axis='x', labelsize=20) 
    ax.tick_params(axis='y', labelsize=20) 
    ax.set_ylim(15, 105)  
    ax.set_yticks(np.arange(20, 101, 20)) 

def generate_p_value_modification_subplot(filename, attack, ax, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100, seeds=[42]):
    llm_config = load_llm_config("meta-llama/Llama-3.2-3B")
    prompts = load_prompts(filename=filename)
    num_prompts = len(prompts)
    modifications = np.array(modification_values)
    num_modifications = len(modification_values)

    # Dictionary to store p-values for each method
    p_values = {
        "SimMark": np.zeros(num_modifications),
        "Unigram": np.zeros(num_modifications),
        "SoftRedList": np.zeros(num_modifications),
        "ExpMin": np.zeros(num_modifications),
        "SynthID": np.zeros(num_modifications)
    }
    method_names = ["SimMark", "Unigram", "SoftRedList", "ExpMin", "SynthID"]
    methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid"]

    for i, num_modify in enumerate(modification_values):

        # Compute p-values for each method
        threshold = 1e-2

        for method_name, method in zip(method_names, methods):
            results = np.empty((0, num_prompts))
            for seed in seeds:
                new_data = np.array(test_watermark(
                    prompts, num_tokens, llm_config, method, method, f"{attack}_{num_modify}", seed=seed
                ))
                results = np.vstack([results, new_data])
            p_values[method_name][i] = np.mean(results<threshold)*100

    save_filename = f"figures/p_value_vs_{attack}_attack_k{k}_b{b}.pdf"
    if attack=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack=="modify":
        xlabel="Number of Unrelated Token Substitutions"
    elif attack=="translate":
        xlabel="Number of Related Token Substitutions"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    subplot_p_value_modifications(modifications, p_values, ax, xlabel)

def plot_modification_comparison(filename, attacks, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100, seeds=[42]):
    plt.style.use(['science', 'no-latex'])

    # Create a figure with 3 subplots horizontally
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

    for i, attack in enumerate(attacks):
        generate_p_value_modification_subplot(filename, attack, axs[i], k=k, b=b, modification_values=modification_values, num_tokens=num_tokens, seeds=seeds)

    handles, labels = axs[0].get_legend_handles_labels()

    # Remove legends from individual subplots
    for ax in axs:
        ax.legend_.remove()

    # Add a single legend below the plots
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=24)

    # Add y-axis label to the leftmost subplot
    axs[0].set_ylabel("Percent with p-value below .01", fontsize=22)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"Figures/p_value_vs_attack_combined_k{k}_b{b}.pdf")
    plt.close()