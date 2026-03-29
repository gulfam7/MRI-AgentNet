import torch
import numpy as np
from meta_learning import MetaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaModel(input_dim=15, hidden_dim=64, output_dim=3).to(device)
model.load_state_dict(torch.load("meta_model_best.pth", map_location=device))
model.eval()
correction_model_mapping = {
    0: "CycleGAN (Motion Correction)",
    1: "CycleGAN (Denoising)",
    2: "CycleGAN (Reconstruction)"
}
def one_hot_encode(classification, corruption_types=["undersampled", "motion corrupted", "noisy"]):
    one_hot = [0] * len(corruption_types)
    if classification in corruption_types:
        one_hot[corruption_types.index(classification)] = 1
    return one_hot
test_cases = [
    {
        "name": "Perfect Agreement",
        "true_corruption": "motion corrupted",
        "ai1_classification": "motion corrupted",
        "ai2_classification": "motion corrupted",
        "radiologist_classification": "motion corrupted",
        "pi_classification": "motion corrupted"
    },
    {
        "name": "Conflicting AI Models",
        "true_corruption": "undersampled",
        "ai1_classification": "motion corrupted",
        "ai2_classification": "noisy",
        "radiologist_classification": "undersampled",
        "pi_classification": "undersampled"
    },
    {
        "name": "Radiologist Mistake",
        "true_corruption": "noisy",
        "ai1_classification": "noisy",
        "ai2_classification": "noisy",
        "radiologist_classification": "motion corrupted",
        "pi_classification": "noisy"
    },
    {
        "name": "PI Uncertain",
        "true_corruption": "undersampled",
        "ai1_classification": "undersampled",
        "ai2_classification": "motion corrupted",
        "radiologist_classification": "motion corrupted",
        "pi_classification": "motion corrupted"
    },
    {
        "name": "AI Models Make Mistakes",
        "true_corruption": "noisy",
        "ai1_classification": "motion corrupted",
        "ai2_classification": "undersampled",
        "radiologist_classification": "noisy",
        "pi_classification": "noisy"
    }
]
print("\n🚀 Running Edge Case Tests...\n")
for case in test_cases:
    print(f"🧪 **Test Case: {case['name']}**")
    input_vector = (
            one_hot_encode(case["true_corruption"]) +
            one_hot_encode(case["ai1_classification"]) +
            one_hot_encode(case["ai2_classification"]) +
            one_hot_encode(case["radiologist_classification"]) +
            one_hot_encode(case["pi_classification"])
    )
    input_tensor = torch.tensor([input_vector], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    predicted_model = correction_model_mapping[predicted_label]
    print(f"✅ **Predicted Correction Model:** {predicted_model}")
    print(f"🔹 True Corruption Type: {case['true_corruption']}")
    print(f"🔹 AI1: {case['ai1_classification']}, AI2: {case['ai2_classification']}")
    print(f"🔹 Radiologist: {case['radiologist_classification']}, PI: {case['pi_classification']}")
    print("-------------------------------------------------\n")
print("🎯 **Edge Case Testing Completed!**")
