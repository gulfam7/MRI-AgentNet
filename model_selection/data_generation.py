import json
import random
import numpy as np
random.seed(42)
corruption_types = ["undersampled", "motion corrupted", "noisy"]
corruption_index = {corruption: i for i, corruption in enumerate(corruption_types)}
correction_mapping = {
    "undersampled": "CycleGAN (Reconstruction)",
    "motion corrupted": "CycleGAN (Motion Correction)",
    "noisy": "CycleGAN (Denoising)"
}
def one_hot_encode(classification):
    one_hot = [0] * len(corruption_types)
    one_hot[corruption_index[classification]] = 1
    return one_hot
def generate_synthetic_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        true_corruption = random.choice(corruption_types)
        ai1_classification = random.choice(corruption_types)
        ai2_classification = random.choice(corruption_types)
        radiologist_classification = ai1_classification if random.random() < 0.6 else true_corruption
        pi_classification = true_corruption if random.random() < 0.9 else radiologist_classification
        final_model = correction_mapping[pi_classification]
        sample = {
            "true_corruption": one_hot_encode(true_corruption),
            "ai1_classification": one_hot_encode(ai1_classification),
            "ai2_classification": one_hot_encode(ai2_classification),
            "radiologist_classification": one_hot_encode(radiologist_classification),
            "pi_classification": one_hot_encode(pi_classification),
            "final_model": final_model
        }
        data.append(sample)
    return data
num_samples = 2000
synthetic_data = generate_synthetic_data(num_samples)
with open("meta_training_data.json", "w") as f:
    json.dump(synthetic_data, f, indent=4)
print(f"✅ Successfully generated {num_samples} training samples.")
