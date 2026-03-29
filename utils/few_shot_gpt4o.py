import os

from openai import OpenAI

class GPT4oFewShot:
    def __init__(self):
        """
        Initializes the GPT-4o interface with an explicit OpenAI API key.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def get_few_shot_prompt(self):
        """
        Constructs a few-shot multimodal prompt for GPT-4o.
        These examples prime the model with MRI corruption classifications.
        """
        return [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://www.dropbox.com/scl/fi/tti26ukmr27y8rtwk8nus/608.png?rlkey=fn46e0ejd19p8fz78f0vctg92&st=tleid5du&dl=1"}},
                {"type": "text", "text": "This MRI scan exhibits motion artifacts due to patient movement. The blurring and streaking are visible along the edges."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Motion Corrupted"}
            ]},

            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://www.dropbox.com/scl/fi/o2xrwxbxmrqvqiyal49c6/103.png?rlkey=jsoosp7e7dpg90cvclldh4519&st=pj2dhcuo&dl=1"}},
                {"type": "text", "text": "This MRI scan has aliasing artifacts caused by undersampling. The repetitive patterns and loss of resolution are noticeable."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Undersampled"}
            ]},

            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://www.dropbox.com/scl/fi/u45y588tauanlx71iy3lb/602.png?rlkey=wwu9wigpwfl0zu75jqez9il7b&st=75yqztvv&dl=1"}},
                {"type": "text", "text": "This MRI scan contains high-frequency noise, reducing fine details and affecting visibility."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Noisy"}
            ]}
        ]

    def get_radiologist_prompt(self, image_url, classification_results):
        """
        Generates the actual classification prompt for the radiologist after few-shot learning.
        This is where the user-selected MRI image is evaluated.
        """
        return [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},                         
                {"type": "text", "text": f"""
                    You are a Radiologist specializing in MRI corruption detection. Your task is to **independently classify the corruption** in this MRI scan before reviewing AI model classifications.

                    ### 🔍 Step 1: Your Independent Classification
                    - Classification:  [Motion Corrupted / Undersampled / Noisy / Other]  
                    - Confidence Score (0-1):** [Your confidence level]  
                    - Reasoning: [Explain your classification decision]

                    ### 🤖 Step 2: Compare with AI Model Classifications
                    Below are the two AI assistant classifications:

                    1️⃣ {classification_results[0]["model"]}: {classification_results[0]["classification"]}  
                       - Confidence: {classification_results[0]["confidence"]}  
                       - Justification: {classification_results[0]["reasoning"]}  

                    2️⃣ {classification_results[1]["model"]}: {classification_results[1]["classification"]}  
                       - Confidence: {classification_results[1]["confidence"]}  
                       - Justification: {classification_results[1]["reasoning"]}  

                    ### ✅ Step 3: Final Evaluation
                    - Do the assistants' classifications align with expert MRI knowledge?
                      [Yes / No / Partially]  
                    - Final Agreement Decision: [Agree / Disagree / Need More Evidence]  
                    - Final Confidence Score: [0-1]  
                    - Final Justification: [Explain your agreement or disagreement]  
                    - Recommended Correction Model:  
                      [CycleGAN for MRI Denoising / CycleGAN for MRI Motion Correction / CycleGAN for MRI Reconstruction]
                """}
            ]}
        ]

    def get_few_shot_response(self, image_url, classification_results):
        """
        Sends a few-shot learning request to GPT-4o for MRI corruption classification.
        - **Priming GPT-4o** with MRI examples.
        - **Providing the user MRI for classification**.
        """
        few_shot_prompt = self.get_few_shot_prompt()                                    
        radiologist_prompt = self.get_radiologist_prompt(image_url, classification_results)                 

        final_prompt = few_shot_prompt + radiologist_prompt                     

        try:
            response = self.client.chat.completions.create(

                model="gpt-4o-2024-11-20",
                messages=final_prompt
            )

            classification = response.choices[0].message.content
            return classification

        except Exception as e:
            print(f"❌ Error generating few-shot response from GPT-4o: {e}")
            return None
