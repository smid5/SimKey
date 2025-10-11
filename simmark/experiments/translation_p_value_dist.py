# plots the distribution of the cost of each generation type and its translation-attacked text
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, cbcolors, linestyles, METHODS, KEYS

def plot_p_value_dist_translation(method_name, key_name, num_tokens, filename, k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
    detection_name = method  # Using the same method for detection

    p_values = {}

    # Generate without watermark, no attack, and detection
    p_values['No Watermark'] = [test_watermark(
        prompts, num_tokens, llm_config, "nomark", detection_name, seed=seed
    ) for seed in seeds]

    # Generate with simhash watermark, no attack, and detection
    p_values[f"{method_name} ({key_name})"] = [test_watermark(
        prompts, num_tokens, llm_config, method, detection_name, seed=seed
    ) for seed in seeds]

    # Generate with watermark, with attack, and detection
    p_values[f'{method_name} ({key_name}) + Translation'] = [test_watermark(
        prompts, num_tokens, llm_config, method, detection_name, "translate", seed=seed
    ) for seed in seeds]

    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("log")

    threshold = 1e-40  # Exclude values below this
    for idx, key in enumerate(p_values):
        filtered_p_values = np.array(p_values[key])
        filtered_p_values = filtered_p_values[filtered_p_values > threshold]  # Remove small values

        if len(filtered_p_values) > 0:
            sns.kdeplot(filtered_p_values, 
                        label=key, 
                        log_scale=True, 
                        linewidth=2, 
                        color=cbcolors[idx], 
                        linestyle=linestyles[idx],
                        cut=0)
    
    plt.xlim(1e-15, 1)
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f"Figures/translation_p_val_dist_{method}_{num_tokens}.pdf")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product

def plot_all_p_value_dists_translation(
    num_tokens,
    filename,
    method_names=["ExpMin", "SynthID", "WaterMax"],
    key_names=["SimKey", "Standard"],
    k=4,
    b=4,
    seeds=[42],
    model_name='meta-llama/Meta-Llama-3-8B',
    savepath="Figures/translation_p_val_grid.pdf"
):
    """
    Create a 2x3 grid of p-value distributions for all combinations of
    method_names and key_names.
    """
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    threshold = 1e-40  # Exclude extremely small values

    plt.style.use(['science', 'no-latex'])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.T.flatten()

    for ax, (method_name, key_name) in zip(axes, product(method_names, key_names)):
        method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
        detection_name = method

        p_values = {
            'No Watermark': [
                test_watermark(prompts, num_tokens, llm_config,
                               "nomark", detection_name, seed=seed)
                for seed in seeds
            ],
            f"{method_name} ({key_name})": [
                test_watermark(prompts, num_tokens, llm_config,
                               method, detection_name, seed=seed)
                for seed in seeds
            ],
            f"{method_name} ({key_name}) +\n  Translation": [
                test_watermark(prompts, num_tokens, llm_config,
                               method, detection_name, "translate", seed=seed)
                for seed in seeds
            ]
        }

        for idx, key in enumerate(p_values):
            filtered_p = np.array(p_values[key])
            filtered_p = filtered_p[filtered_p > threshold]
            if len(filtered_p) > 0:
                sns.kdeplot(
                    filtered_p,
                    label=key,
                    log_scale=True,
                    linewidth=3,
                    color=cbcolors[idx],
                    linestyle=linestyles[idx],
                    cut=0,
                    ax=ax
                )

        ax.set_xscale("log")
        ax.set_xlim(1e-15, 1)
        # Remove per-subplot labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=12) 
        ax.legend(fontsize=12)

    # Turn off any unused subplots
    for extra_ax in axes[len(method_names)*len(key_names):]:
        extra_ax.axis("off")

    # Shared axis labels for the whole figure
    fig.supxlabel(r"$p$-value", fontsize=20)
    fig.supylabel("Frequency", fontsize=20)

    # plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # leave space for shared labels
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)