import os
import requests
from PIL import Image
import google.generativeai as genai
import io


class GeminiFewShotInterface:
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
        response = self._generate_response(image_path, prompt)
        return response

    def get_few_shot_response(self, few_shot_prompt):
        """
        Handles **few-shot learning prompts** by sending multiple examples + query.

        Parameters:
        - few_shot_prompt: List of user-model exchanges formatted as Gemini expects.

        Returns:
        - Response from Gemini containing classification, reasoning, and confidence.
        """
        try:
            response = self.model.generate_content(few_shot_prompt)
            if response.text:
                return response.text.strip()

        except Exception as e:
            print(f"Error generating few-shot response from Gemini: {e}")

        return None

    def _generate_response(self, few_shot_prompt):
        try:

            if not isinstance(few_shot_prompt, list):
                few_shot_prompt = [few_shot_prompt]


            response = self.model.generate_content(few_shot_prompt)

            if response.text:
                return response.text.strip()

        except Exception as e:
            print(f"Error generating few-shot response from Gemini: {e}")

        return None
