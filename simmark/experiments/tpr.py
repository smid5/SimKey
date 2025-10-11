import numpy as np
from .utils import load_llm_config, load_prompts, test_watermark, METHODS, KEYS

def compute_tpr(pvals_unwm, pvals_wm, fpr):
    pvals_unwm = np.asarray(pvals_unwm).flatten()
    pvals_wm = np.asarray(pvals_wm).flatten()
    # Step 1: Find threshold tau from unwatermarked p-values
    sorted_unwm = np.sort(pvals_unwm)
    print(len(sorted_unwm))
    k = int(np.floor(fpr * len(sorted_unwm)))
    tau = sorted_unwm[k]

    # Step 2: Compute TPR using the same threshold
    tpr = np.mean(pvals_wm <= tau)
    return tpr, tau

def get_tprs(method_name, key_name, fprs, num_tokens, filename, attack_name="", k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
    detection_name = method

    # Generate without watermark
    pvals_unwm = np.concatenate([test_watermark(prompts, num_tokens, llm_config, "nomark", detection_name, seed=seed) 
                  for seed in seeds])

    # Generate with watermark
    pvals_wm = np.concatenate([test_watermark(prompts, num_tokens, llm_config, method, detection_name, attack_name=attack_name, seed=seed)
                for seed in seeds])

    for fpr in fprs:
        tpr, tau = compute_tpr(pvals_unwm, pvals_wm, fpr)
        print(f"Method: {method_name}, Key: {key_name}, FPR: {fpr}, TPR: {tpr}, Tau: {tau}")