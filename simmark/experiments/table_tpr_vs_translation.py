import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, METHODS, COLORS, KEYS, LINESTYLES, moving_average
from collections import defaultdict
from .tpr import compute_tpr

def generate_tpr_translation_table(num_tokens, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["SimKey", "Standard"], fpr= 1e-2, k=4, b=4, wm_seeds=[42], unwm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    tprs = defaultdict(dict)

    for method_name in method_names:
        for key_name in key_names:
            method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
            pvals_unwm = [test_watermark(
                prompts, num_tokens, llm_config, "nomark", method, seed=seed
            ) for seed in unwm_seeds]
            pvals_translated = [test_watermark(
                prompts, num_tokens, llm_config, method, method, "translate", seed=seed
            ) for seed in wm_seeds]
            
            tpr, _ = compute_tpr(pvals_unwm, pvals_translated, fpr)
            tprs[(method_name, key_name)] = tpr
            print(f"Method: {method_name}, Key: {key_name}, TPR under translation at FPR {fpr}: {tpr}")
    log_filename = f"logs/tpr_vs_translate_k{k}_b{b}.txt"

    with open(log_filename, "a") as f:
        for (method_name, key_name), tpr_values in tprs.items():
            f.write(f"Method: {method_name}, Key: {key_name}, TPR under translation at FPR {fpr}: {tpr_values}\n")
        print("\n")