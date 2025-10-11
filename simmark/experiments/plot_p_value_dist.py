# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_watermark, load_prompts, cbcolors, linestyles, METHODS

def plot_p_value_dist(method_name, num_tokens, filename, k=4, b=4):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    method = f"simmark_{k}_{b}" if method_name == "SimMark" else METHODS[method_name]
    detection_name = method

    p_values = {}

    # Generate without watermark, no attack, and detection
    p_values['No Watermark'] = test_watermark(
        prompts, num_tokens, llm_config, "nomark", detection_name
    )

    # Generate with simhash watermark, no attack, and detection
    p_values[method_name] = test_watermark(
        prompts, num_tokens, llm_config, method, detection_name
    )

    # Generate with simhash watermark, with attack, and detection
    p_values[f'{method_name} + Attack 1'] = test_watermark(
        prompts, num_tokens, llm_config, method, detection_name, "modify_1"
    )

    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("log")
    for idx, key in enumerate(p_values):
        sns.kdeplot(p_values[key], label=key, log_scale=True, linewidth=2, color=cbcolors[idx], linestyle=linestyles[idx], cut=0)
    
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f"Figures/p_val_dist_{method}_{num_tokens}.pdf")

    # Show the plot