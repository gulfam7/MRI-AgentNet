import re
import logging


class PlanParser:
    def parse_plan(self, gpt_response):
        """
        Parses GPT-4o's response to extract classification, confidence, recommended model, and correction steps.

        Parameters:
        - gpt_response: String containing GPT-4o's response.

        Returns:
        - plan_details: Dictionary with keys 'classification', 'confidence', 'recommended_model', and 'plan'.
        """
        plan_details = {
            'classification': 'Unknown',
            'confidence': 0.0,
            'recommended_model': 'Unknown',
            'plan': []
        }


        no_corruption_match = re.search(r'Classification:\s*no corruption', gpt_response, re.IGNORECASE)
        if no_corruption_match:
            plan_details['classification'] = 'no_corruption'

            confidence_match = re.search(r'Confidence:\s*([0-1](?:\.\d+)?)', gpt_response, re.IGNORECASE)
            if confidence_match:
                plan_details['confidence'] = float(confidence_match.group(1))

            plan_details['recommended_model'] = 'no_correction_required'

            plan_details['plan'] = []
            logging.info("No corruption detected.")
            print("No corruption detected.")
            return plan_details


        corruption_match = re.search(r'Classification:\s*(motion corrupted|undersampled|noisy)', gpt_response,
                                     re.IGNORECASE)
        if corruption_match:
            corruption_type = corruption_match.group(1).lower().replace(' ', '_')
            plan_details['classification'] = corruption_type

            confidence_match = re.search(r'Confidence:\s*([0-1](?:\.\d+)?)', gpt_response, re.IGNORECASE)
            if confidence_match:
                plan_details['confidence'] = float(confidence_match.group(1))

            recommended_model_match = re.search(r'Recommended Model:\s*(.*)', gpt_response, re.IGNORECASE)
            if recommended_model_match:
                recommended_model = recommended_model_match.group(1).strip().lower()
                model_mapping = {
                    'cyclegan for mri denoising': 'denoising',
                    'pix2pix for mri motion correction': 'motion_correction',

                }
                mapped_model = model_mapping.get(recommended_model, 'Unknown')
                plan_details['recommended_model'] = mapped_model
            else:
                logging.warning("No recommended model found in GPT-4o response.")
                print("No recommended model found in GPT-4o response.")


            plan_match = re.search(r'Plan:\s*(.*)', gpt_response, re.IGNORECASE | re.DOTALL)
            if plan_match:
                plan_text = plan_match.group(1).strip()

                plan_steps = re.findall(r'\d+\.\s*(.*)', plan_text)
                plan_details['plan'] = [step.strip() for step in plan_steps if step.strip()]
                logging.info(f"Plan steps: {plan_details['plan']}")
                print(f"Plan steps: {plan_details['plan']}")
            else:
                logging.warning("No plan found in GPT-4o response.")
                print("No plan found in GPT-4o response.")
        else:
            logging.warning("No known corruption type detected in GPT-4o response.")
            print("No known corruption type detected in GPT-4o response.")

        return plan_details
