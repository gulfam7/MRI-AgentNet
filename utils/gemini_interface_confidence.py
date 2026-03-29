import os
import google.generativeai as genai
from PIL import Image
import io
import requests


class GeminiInterface:
    def __init__(self, model):
        """
        Initializes the Gemini API interface.
        """
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def get_plan(self, image_path, prompt):
        """
        Generates a **corruption classification** for an MRI image using Gemini.

        Parameters:
        - image_path: Path to the MRI image file.
        - prompt: Instruction prompt for Gemini.

        Returns:
        - Response from Gemini containing classification, recommended model, correction plan, and confidence score.
        """
        prompt += (
            "\nPlease analyze the MRI image and **explicitly include** the following components in your response:\n"
            "1. Classification** of the corruption (e.g., motion corrupted, undersampled, noisy, or no corruption).\n"
            "2. A recommended model to address the corruption:\n"
            "   - CycleGAN for motion correction\n"
            "   - CycleGAN for MRI denoising\n"
            "   - CycleGAN for MRI reconstruction\n"
            "3. A step-by-step correction plan.\n"
            "4. A confidence score in this format: 'Confidence Score: [0-1]'.\n"
            "Ensure all these components are included explicitly.\n"
            "Do not use newline characters between numbered sections"
        )

        response = self._generate_response(image_path, prompt)
        return response

    def get_initial_classification(self, image_path, prompt):
        """
        Classifies whether the MRI image is **image space** or **k-space**.

        Parameters:
        - image_path: Path to the MRI image file.
        - prompt: Instruction prompt for Gemini.

        Returns:
        - A string: 'image space' or 'k-space'.
        """
        prompt += (
            "\nAnalyze the given MRI image and determine whether it is **image space** or **k-space**."
            "Provide a **single-word** response: 'image space' or 'k-space'."
        )

        response = self._generate_response(image_path, prompt)
        if response:
            return response.strip().lower()
        return "Unknown"

    def _generate_response(self, image_url, prompt):
        try:

            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                print(f"Error: Unable to download image from Dropbox. Status Code: {response.status_code}")
                return None


            temp_image_path = "temp_mri_image.png"
            with open(temp_image_path, "wb") as f:
                f.write(response.content)


            image = Image.open(temp_image_path)


            response = self.model.generate_content([prompt, image])
            if response.text:
                return response.text.strip()

        except Exception as e:
            print(f"Error generating response from Gemini: {e}")

        return None

    def get_agent_plan(self, image_path, prompt):
        """
        Generates a **corruption classification** for an MRI image using Gemini.

        Parameters:
        - image_path: Path to the MRI image file.
        - prompt: Instruction prompt for Gemini.

        Returns:
        - Response from Gemini containing classification, recommended model, correction plan, and confidence score.
        """

        response = self._generate_response(image_path, prompt)
        return response
