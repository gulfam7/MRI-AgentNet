def get_few_shot_prompt():
    """
    Generates a structured few-shot prompt by sequentially providing examples.
    """
    few_shot_examples = []


    few_shot_examples.append(
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": "https://www.dropbox.com/scl/fi/tti26ukmr27y8rtwk8nus/608.png?dl=1"}},
            {"text": "This MRI scan exhibits motion artifacts characterized by blurring and ghosting due to patient movement."}
        ]})
    few_shot_examples.append({"role": "model", "parts": [{"text": "Motion corrupted MRI due to patient movement."}]})


    few_shot_examples.append(
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": "https://www.dropbox.com/scl/fi/io0e2l3sg4yaqte1eikii/101.png?dl=1"}},
            {"text": "This MRI scan has aliasing artifacts caused by undersampling."}
        ]})
    few_shot_examples.append({"role": "model", "parts": [{"text": "Undersampling artifact in MRI causing aliasing distortions."}]})


    few_shot_examples.append(
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": "https://www.dropbox.com/scl/fi/olfp0arkxrzldfhigrq3v/555.png?dl=1"}},
            {"text": "High-frequency noise affecting MRI clarity."}
        ]})
    few_shot_examples.append({"role": "model", "parts": [{"text": "Significant noise reducing MRI visibility."}]})

    return few_shot_examples


def get_radiologist_prompt(image_url, classification_results):
    """
    Generates a **separate** classification prompt for the radiologist.
    This is where the user-selected MRI image is processed.
    """
    radiologist_prompt = [
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": image_url}},                            
            {"text": f"""
                You are a **Radiologist specializing in MRI corruption detection**. Your task is to **independently classify the corruption** in this MRI scan before reviewing AI model classifications.

                ### 🔍 **Step 1: Your Independent Classification**
                - **What type of corruption is present?**  
                  [Motion Corrupted / Undersampled / Noisy / Other]  
                - **Confidence Score (0-1):** [Your confidence level]  
                - **Justification:** [Explain your classification decision]

                ### 🤖 **Step 2: Compare with AI Model Classifications**
                Below are the two AI assistant classifications:

                1️⃣ **{classification_results[0]["model"]}:** {classification_results[0]["classification"]}  
                   - Confidence: {classification_results[0]["confidence"]}  
                   - Justification: {classification_results[0]["reasoning"]}  

                2️⃣ **{classification_results[1]["model"]}:** {classification_results[1]["classification"]}  
                   - Confidence: {classification_results[1]["confidence"]}  
                   - Justification: {classification_results[1]["reasoning"]}  

                ### ✅ **Step 3: Final Evaluation**
                - **Do the assistants' classifications align with expert MRI knowledge?**  
                  [Yes / No / Partially]  
                - **Final Agreement Decision:** [Agree / Disagree / Need More Evidence]  
                - **Final Confidence Score:** [0-1]  
                - **Final Justification:** [Explain your agreement or disagreement]  
                - **Recommended Correction Model:**  
                  [CycleGAN for MRI Denoising / CycleGAN for MRI Motion Correction / CycleGAN for MRI Reconstruction]
                """}
        ]}
    ]

    return radiologist_prompt
