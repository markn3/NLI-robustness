import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
confusion_matrices = {level: np.zeros((3, 3), dtype=int) for level in noise_levels}

def load_confusion_matrix(jsonl_file, noise_level, confusion_matrices):
    """
    Reads a JSONL file and populates the confusion matrix for a given noise level.
    
    Parameters:
    - jsonl_file (str): Path to the JSONL file containing predictions and labels.
    - noise_level (float): The noise level associated with this file.
    - confusion_matrices (dict): Dictionary to store confusion matrices for different noise levels.
    """
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            true_label = data["label"]
            predicted_label = data["predicted_label"]
            confusion_matrices[noise_level][true_label][predicted_label] += 1

confusion_matrices_base = {level: np.zeros((3, 3), dtype=int) for level in noise_levels}
confusion_matrices_20 = {level: np.zeros((3, 3), dtype=int) for level in noise_levels}
confusion_matrices_40 = {level: np.zeros((3, 3), dtype=int) for level in noise_levels}

load_confusion_matrix('eval_base_output\\0_0\\eval_predictions.jsonl', noise_level=0.0, confusion_matrices=confusion_matrices_base)
load_confusion_matrix('eval_base_output\\0_2\\eval_predictions.jsonl', noise_level=0.2, confusion_matrices=confusion_matrices_base)
load_confusion_matrix('eval_base_output\\0_4\\eval_predictions.jsonl', noise_level=0.4, confusion_matrices=confusion_matrices_base)
load_confusion_matrix('eval_base_output\\0_6\\eval_predictions.jsonl', noise_level=0.6, confusion_matrices=confusion_matrices_base)
load_confusion_matrix('eval_base_output\\0_8\\eval_predictions.jsonl', noise_level=0.8, confusion_matrices=confusion_matrices_base)
load_confusion_matrix('eval_base_output\\1_0\\eval_predictions.jsonl', noise_level=1.0, confusion_matrices=confusion_matrices_base)

load_confusion_matrix('eval_aug_output_20\\0_0\\eval_predictions.jsonl', noise_level=0.0, confusion_matrices=confusion_matrices_20)
load_confusion_matrix('eval_aug_output_20\\0_2\\eval_predictions.jsonl', noise_level=0.2, confusion_matrices=confusion_matrices_20)
load_confusion_matrix('eval_aug_output_20\\0_4\\eval_predictions.jsonl', noise_level=0.4, confusion_matrices=confusion_matrices_20)
load_confusion_matrix('eval_aug_output_20\\0_6\\eval_predictions.jsonl', noise_level=0.6, confusion_matrices=confusion_matrices_20)
load_confusion_matrix('eval_aug_output_20\\0_8\\eval_predictions.jsonl', noise_level=0.8, confusion_matrices=confusion_matrices_20)
load_confusion_matrix('eval_aug_output_20\\1_0\\eval_predictions.jsonl', noise_level=1.0, confusion_matrices=confusion_matrices_20)

load_confusion_matrix('eval_aug_output_40\\0_0\\eval_predictions.jsonl', noise_level=0.0, confusion_matrices=confusion_matrices_40)
load_confusion_matrix('eval_aug_output_40\\0_2\\eval_predictions.jsonl', noise_level=0.2, confusion_matrices=confusion_matrices_40)
load_confusion_matrix('eval_aug_output_40\\0_4\\eval_predictions.jsonl', noise_level=0.4, confusion_matrices=confusion_matrices_40)
load_confusion_matrix('eval_aug_output_40\\0_6\\eval_predictions.jsonl', noise_level=0.6, confusion_matrices=confusion_matrices_40)
load_confusion_matrix('eval_aug_output_40\\0_8\\eval_predictions.jsonl', noise_level=0.8, confusion_matrices=confusion_matrices_40)
load_confusion_matrix('eval_aug_output_40\\1_0\\eval_predictions.jsonl', noise_level=1.0, confusion_matrices=confusion_matrices_40)

# Calculate misclassification rates for each model
def calculate_misclassification_rates(confusion_matrices):
    misclassification_rates = []
    for noise_level in noise_levels:
        confusion_matrix = confusion_matrices[noise_level]
        rates = []
        for i in range(3):
            total_actual = confusion_matrix[i].sum()
            for j in range(3):
                if i != j:
                    misclassified_count = confusion_matrix[i][j]
                    misclassification_rate = misclassified_count / total_actual if total_actual > 0 else 0
                    rates.append(misclassification_rate)
        misclassification_rates.append(rates)
    return np.array(misclassification_rates)

misclassification_data_base = calculate_misclassification_rates(confusion_matrices_base)
misclassification_data_20 = calculate_misclassification_rates(confusion_matrices_20)
misclassification_data_40 = calculate_misclassification_rates(confusion_matrices_40)

vmin = min(misclassification_data_base.min(), misclassification_data_20.min(), misclassification_data_40.min())
vmax = max(misclassification_data_base.max(), misclassification_data_20.max(), misclassification_data_40.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Base model heatmap
sns.heatmap(misclassification_data_base, annot=True, cmap="YlOrBr", vmin=vmin, vmax=vmax, 
            xticklabels=["Entail→Neut.", "Entail→Cont.", "Neut.→Entail", "Neut.→Cont.", "Cont.→Entail", "Cont.→Neut."],
            yticklabels=noise_levels, ax=axes[0])
axes[0].set_title("Base Model")
axes[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better fit


# 20% Noise Inoculated model heatmap
sns.heatmap(misclassification_data_20, annot=True, cmap="YlOrBr", vmin=vmin, vmax=vmax, 
            xticklabels=["Entail→Neut.", "Entail→Cont.", "Neut.→Entail", "Neut.→Cont.", "Cont.→Entail", "Cont.→Neut."],
            yticklabels=noise_levels, ax=axes[1])
axes[1].set_title("20% Noise Inoculated Model")
axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better fit


# 40% Noise Inoculated model heatmap
sns.heatmap(misclassification_data_40, annot=True, cmap="YlOrBr", vmin=vmin, vmax=vmax, 
            xticklabels=["Entail→Neut.", "Entail→Cont.", "Neut.→Entail", "Neut.→Cont.", "Cont.→Entail", "Cont.→Neut."],
            yticklabels=noise_levels, ax=axes[2])
axes[2].set_title("40% Noise Inoculated Model")
axes[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better fit


plt.suptitle("Comparison of Misclassification Rates Across Noise Levels for Different Models")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()