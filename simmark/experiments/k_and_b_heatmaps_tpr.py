import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_llm_config, test_watermark, load_prompts, METHODS, KEYS
from .tpr import compute_tpr
import matplotlib.colors as mcolors

def plot_heatmap(data, k_values, b_values, title, filename):
    plt.figure(figsize=(6, 5))

    # Avoid log(0) errors by setting a small lower bound
    data = np.maximum(data, 1e-20)

    sns.heatmap(data, annot=True, fmt=".1e", cmap="coolwarm", norm=mcolors.LogNorm(vmin=data.min(), vmax=data.max()), 
                xticklabels=b_values, yticklabels=k_values)

    # sns.heatmap(data, annot=True, fmt=".1e", cmap="coolwarm", xticklabels=b_values, yticklabels=k_values)
    plt.xlabel("b")
    plt.ylabel("k")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_tpr_heatmaps(k_values, b_values, method_name, key_name, num_tokens, filename, unwm_seeds=[42], wm_seeds=[42], fpr=1e-2, model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    
    tprs = np.zeros((len(k_values), len(b_values)))
    
    for i, k in enumerate(k_values):
        for j, b in enumerate(b_values):
            method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
            
            p_values_unwatermarked = [
                test_watermark(prompts, num_tokens, llm_config, "nomark", method, seed=seed)
                for seed in unwm_seeds
            ]

            p_values_translated = [
                test_watermark(prompts, num_tokens, llm_config, method, method, "translate", seed=seed)
                for seed in wm_seeds
            ]

            tpr, _ = compute_tpr(p_values_unwatermarked, p_values_translated, fpr)
            tprs[i, j] = tpr

    plot_heatmap(tprs, k_values, b_values, "TPRs for Translated Text", f"Figures/heatmap_tpr_{method_name}_{key_name}_translated.pdf")