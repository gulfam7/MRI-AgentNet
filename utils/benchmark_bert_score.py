import torch
from bert_score import score


BENCHMARK_JUSTIFICATIONS = {
    "motion corrupted": "This MRI scan exhibits characteristic motion artifacts, which include blurring and ghosting of anatomical structures along the phase-encoding direction. These distortions occur due to patient movement during acquisition, leading to repeated signal misalignment across adjacent slices. Motion artifacts often disrupt fine structural details and can obscure pathology, making correction essential.",
    "undersampled": "This MRI scan displays aliasing artifacts due to k-space undersampling. The repetitive wrap-around patterns at the image periphery and loss of spatial resolution are typical of accelerated acquisitions where not enough frequency data was collected. These artifacts can cause structural distortions, particularly in high-contrast regions, leading to compromised image quality.",
    "noisy": "This MRI scan contains significant high-frequency noise, leading to granular texture and reduced visibility of fine anatomical details. The low SNR results in excessive random signal variations, making it difficult to distinguish structures with subtle contrast. Noise can be amplified by acquisition settings such as low field strength or insufficient averaging, requiring denoising techniques to improve clarity."
}


def compute_benchmark_bert_score(evaluator_justifications, corruption_type):
    """
    Computes BERTScore for the evaluator justifications against the expert benchmark justification.

    Parameters:
    - evaluator_justifications: List of justifications (Assistant, Radiologist, PI).
    - corruption_type: The predicted MRI corruption type (motion corrupted, undersampled, noisy).

    Returns:
    - bert_scores: List of BERTScores for each evaluator against the benchmark.
    """
    if corruption_type not in BENCHMARK_JUSTIFICATIONS:
        print(f"⚠️ Warning: Unknown corruption type '{corruption_type}', skipping BERTScore computation.")
        return None

    reference_text = [BENCHMARK_JUSTIFICATIONS[corruption_type]] * len(evaluator_justifications)

    P, R, F1 = score(evaluator_justifications, reference_text, lang="en", model_type="microsoft/deberta-xlarge-mnli")
    return F1.tolist()
