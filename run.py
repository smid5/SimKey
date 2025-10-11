import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from simmark.experiments.sentence_length_vs_p_value import sentence_length_median_pvalue
from simmark.experiments.translation_p_value_dist import plot_p_value_dist_translation, plot_all_p_value_dists_translation
from simmark.experiments.plot_distortion_dist import sentence_length_median_distortion
from simmark.experiments.num_modifications_vs_tpr import generate_tpr_modification_experiment
from simmark.experiments.sentence_length_vs_tpr import sentence_length_tpr
from simmark.experiments.plot_sentence_length_combined import plot_sentence_length_combined
from simmark.experiments.table_tpr_vs_translation import generate_tpr_translation_table
from simmark.experiments.num_modifications_vs_p_value import generate_p_value_modification_experiment
from simmark.experiments.num_modifications_combined import plot_num_modifications_combined
from simmark.experiments.k_and_b_heatmaps_tpr import generate_tpr_heatmaps

filename = 'data/prompts.txt'
num_tokens = 100
seeds = [42, 0]
unwm_seeds = [42, 0, 1]
# model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
model_name='meta-llama/Meta-Llama-3-8B'

length_variations = list(range(25, 105, 5))
sentence_length_median_distortion(length_variations, filename, seeds=seeds, model_name=model_name)
sentence_length_median_pvalue(length_variations, filename, seeds=seeds, model_name=model_name)

method_names = ["ExpMin", "SynthID", "WaterMax"]
modification_values = list(range(0, 31, 3))
generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name="translate", method_names=method_names, seeds=seeds, model_name=model_name)

generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="modify", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name, output_log_file=True)
generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="translate", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)
generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="mask", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name, output_log_file=True)
sentence_length_tpr(length_variations, filename, method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)

plot_all_p_value_dists_translation(num_tokens, filename, seeds=seeds, model_name=model_name)
plot_sentence_length_combined(length_variations, filename, method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)
generate_tpr_translation_table(num_tokens, filename, method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)
plot_num_modifications_combined(modification_values, num_tokens, filename, attack_name="translate", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)

k_values = [2, 4, 6, 8, 10]
b_values = [2, 4, 6, 8, 10]
generate_tpr_heatmaps(k_values, b_values, method_name="ExpMin", key_name="SimKey", num_tokens=num_tokens, filename=filename, unwm_seeds=unwm_seeds, wm_seeds=seeds, fpr=1e-2, model_name=model_name)
generate_tpr_heatmaps(k_values, b_values, method_name="WaterMax", key_name="SimKey", num_tokens=num_tokens, filename=filename, unwm_seeds=unwm_seeds, wm_seeds=seeds, fpr=1e-2, model_name=model_name)
generate_tpr_heatmaps(k_values, b_values, method_name="SynthID", key_name="SimKey", num_tokens=num_tokens, filename=filename, unwm_seeds=unwm_seeds, wm_seeds=seeds, fpr=1e-2, model_name=model_name)