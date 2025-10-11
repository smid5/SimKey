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

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_sentence_length_median_pvalue(sentence_lengths, pvalues, filename):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6, 4))  # wider plot
    MIN_PVAL = 1e-16

    for (label, values) in pvalues.items():
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

        clipped_values = [max(v, MIN_PVAL) for v in values]
        moving_avg = moving_average(clipped_values, window_size=3)
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

    plt.yscale("log")  # Set y-axis to log scale to better capture small variations
    plt.xscale("linear")
    yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, MIN_PVAL]
    plt.yticks(yticks)

    # Format ticks: show "<10^-15" at bottom
    def custom_y_ticks(val, _):
        if val <= MIN_PVAL + 1e-20:  # Allow for float imprecision
            return r"$<10^{-16}$"
        exponent = int(np.log10(val))
        return f"$10^{{{exponent}}}$"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_y_ticks))
    # Labels and ticks
    plt.xlabel("Sentence Length", fontsize=12)
    plt.ylabel("Median p-value", fontsize=12)
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

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def sentence_length_median_pvalue(length_variations, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    p_values = defaultdict(dict)

    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue

        for method_name in method_names:
            for key_name in key_names:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}"

                p_vals = [test_watermark(
                    applicable_prompts, length, llm_config, method, method, seed=seed
                ) for seed in seeds]
                median_pval = np.median(p_vals)
                p_values[(method_name, key_name)][length] = median_pval
        
    sorted_lengths = sorted(p_values[(method_names[0], key_names[0])].keys())
    for key in p_values:
        p_values[key] = [p_values[key][l] for l in sorted_lengths]
    plot_sentence_length_median_pvalue(sorted_lengths, p_values, f"Figures/sentence_length_vs_pvalue.pdf")

def plot_sentence_length_p_values(sentence_lengths, p_values, filename):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(7.5, 4))
    MIN_PVAL = 1e-16
    
    for idx, (label, values) in enumerate(p_values.items()):
        clipped_values = [max(v, MIN_PVAL) for v in values]
        plt.plot(sentence_lengths, clipped_values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)
    
    plt.yscale("log")  # Set y-axis to log scale to better capture small variations
    plt.xscale("linear")
    plt.xlabel("Sentence Length")
    plt.ylabel("Median p-Value")

    yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, MIN_PVAL]
    plt.yticks(yticks)

    # Format ticks: show "<10^-15" at bottom
    def custom_y_ticks(val, _):
        if val <= MIN_PVAL + 1e-20:  # Allow for float imprecision
            return r"$<10^{-16}$"
        exponent = int(np.log10(val))
        return f"$10^{{{exponent}}}$"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_y_ticks))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_sentence_length_p_values(filename, k=4, b=4, length_variations=list(range(25, 105, 5)), seeds=[42]):
    llm_config = load_llm_config("meta-llama/Llama-3.2-3B")
    prompts = load_prompts(filename=filename)
    
    p_values = {"No Watermark": {}, "SimMark": {}, "SoftRedList": {}, "Unigram": {}, "ExpMin": {}, "SynthID": {}}
    generation_methods = ["nomark", f"simmark_{k}_{b}", "softred", "unigram", "expmin", "synthid"]
    detection_methods = [f"simmark_{k}_{b}", f"simmark_{k}_{b}", "softred", "unigram", "expmin", "synthid"]
    
    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue
        
        num_tokens_list = [length - len(p.split()) for p in applicable_prompts]

        for method_name, gen_method, det_method in zip(list(p_values.keys()), generation_methods, detection_methods):
            all_pvals = []
            for seed in seeds:
                new_data = [
                    test_watermark([p], num_tokens, llm_config, gen_method, det_method, seed=seed)[0]
                    for p, num_tokens in zip(applicable_prompts, num_tokens_list)
                ]
                all_pvals.extend(new_data)

            p_values[method_name][length] = np.median(all_pvals)

    sorted_lengths = sorted(p_values["No Watermark"].keys())
    for key in p_values:
        p_values[key] = [p_values[key][l] for l in sorted_lengths]
    
    plot_sentence_length_p_values(sorted_lengths, p_values, f"Figures/sentence_length_vs_p_values_k{k}_b{b}.pdf")

if __name__ == '__main__':
    generate_sentence_length_p_values("sentence_starters.txt")