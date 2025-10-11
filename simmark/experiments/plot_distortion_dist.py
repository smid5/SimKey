# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_distortion, load_prompts, METHODS, COLORS, KEYS, LINESTYLES, moving_average
from collections import defaultdict
import numpy as np

def plot_sentence_length_median_distortion(sentence_lengths, distortions, filename):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(8, 5))  # wider plot

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

    # Labels and ticks
    plt.xlabel("Sentence Length", fontsize=13)
    plt.ylabel("Median Perplexity", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title("Perplexity by Sentence Length", fontsize=18)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(
        fontsize=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # negative y moves legend below axes
        ncol=4
    )

    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def sentence_length_median_distortion(length_variations, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B', plot=True):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    distortions = defaultdict(dict)
    if "No Watermark" not in method_names:
        method_names.append("No Watermark")

    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue

        for method_name in method_names:
            if method_name == "No Watermark":
                method = "nomark"
                detection_name = f"expmin_simhash"
                distortion_vals = [test_distortion(
                    applicable_prompts, length, llm_config, method, detection_name, seed=seed
                ) for seed in seeds]
                median_distortion = np.median(distortion_vals)
                distortions[method_name][length] = median_distortion

            else:
                for key_name in key_names:
                    method = f"{METHODS[method_name]}_{KEYS[key_name]}"

                    distortion_vals = [test_distortion(
                        applicable_prompts, length, llm_config, method, method, seed=seed
                    ) for seed in seeds]
                    median_distortion = np.median(distortion_vals)
                    distortions[(method_name, key_name)][length] = median_distortion
        
    sorted_lengths = sorted(distortions["No Watermark"].keys())
    for key in distortions:
        distortions[key] = [distortions[key][l] for l in sorted_lengths]
    if plot:
        plot_sentence_length_median_distortion(sorted_lengths, distortions, f"Figures/sentence_length_vs_distortion.pdf")
    else:
        return sorted_lengths, distortions

def plot_distortion_dist(num_tokens, filename, k=4, b=4):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    method_names = {"SimMark", "ExpMin", "SoftRedList", "Unigram", "SynthID", "No Watermark"}

    perplexity = {}

    for method_name in method_names:
        method = f"simmark_{k}_{b}" if method_name == "SimMark" else METHODS[method_name]
        detection_name = f"simmark_{k}_{b}" if method == "nomark" else method


        perplexity[method_name]= test_distortion(
            prompts, num_tokens, llm_config, method, detection_name
        )

    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("linear")
    for idx, key in enumerate(perplexity):
        sns.kdeplot(perplexity[key], label=key, log_scale=False, linewidth=2, color=COLORS[key], cut=0)
    
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f"figures/perplexity_dist_{num_tokens}.pdf")

    # Show the plot