import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import tempfile
import uuid
import logging
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
import difflib
import google.generativeai as genai
from utils.gemini_interface_confidence import GeminiInterface
from model_selection.meta_learning import MetaModel
import torch
from utils.bert_score_utils import compute_bert_score
from utils.benchmark_bert_score import compute_benchmark_bert_score
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_few_shot_interface import GeminiFewShotInterface
from utils.gpt4o_interface import GPT4oInterface
from utils.plan_parser import PlanParser
from utils.model_selector import ModelSelector
from openai import OpenAI
from utils import data_processing_confidence
from utils import few_shot_prompt_gemini
from utils import few_shot_gpt4o
from models.options.test_options import TestOptions
from models.test import run_inference
_DISPLAY_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import cv2
    _DISPLAY_AVAILABLE = True
    print("✅ Display libraries (matplotlib, cv2) found.")
except ImportError:
    print("⚠️ WARNING: matplotlib or cv2 not found. Image display will be skipped.")
class MRI_Agent:
    def __init__(self, client,
                 model_image_space="gpt-4o-2024-11-20",
                 model_corruption1="gpt-4o-2024-11-20",
                 model_principal="ft:gpt-4o-2024-08-06:personal:combined-900:Av7pB23x",
                 results_dir_base=r'D:\MRI-AgentNet\models\results',
                 checkpoints_dir_base=r'D:\MRI-AgentNet\models\checkpoints'
                 ):
        self.gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        self.gpt4o_image_space = GPT4oInterface(client=client, model=model_image_space)
        self.gpt4o_corruption = GPT4oInterface(client=client, model=model_corruption1)
        self.gemini_corruption = GeminiInterface(self.gemini_model)
        self.radiologist_evaluator = GPT4oInterface(client=client, model=model_image_space)
        self.principal_evaluator = GPT4oInterface(client=client, model=model_principal)
        self.parser = PlanParser()
        self.model_selector = ModelSelector()
        self.results_dir = results_dir_base
        self.checkpoints_dir = checkpoints_dir_base
        print(f"✅ MRI_Agent Initialized. Results Dir: {self.results_dir}, Checkpoints Dir: {self.checkpoints_dir}")
    def process(self, data_path, prompt):
        print("Processing MRI data...")
        processed_image_path = data_processing_confidence.preprocess_data(data_path)
        if not processed_image_path:
            print("Failed to preprocess MRI data.")
            return
        print(f"Processed image saved at: {processed_image_path}")
        image_url = data_processing_confidence.upload_to_dropbox(processed_image_path)
        if not image_url:
            print("Failed to upload MRI data to Dropbox.")
            return
        print(f"Image uploaded to Dropbox. Accessible at: {image_url}")
        print("Sending pre-classification request to GPT-4o...")
        pre_classification_prompt = (
            f"You are an MRI expert. Analyze the following MRI image and determine whether it is in image space or k-space:\n"
            f"{image_url}\n"
            "Please respond with either 'image space' or 'k-space' only."
        )
        pre_classification_response = self.gpt4o_image_space.get_initial_classification(image_url,
                                                                                        pre_classification_prompt)
        if not pre_classification_response:
            print("Failed to get a response from GPT-4o for pre-classification.")
            return
        print(f"GPT-4o Pre-classification Response: {pre_classification_response}")
        if 'k-space' in pre_classification_response.lower():
            print("GPT-4o classified the data as k-space. Converting original data to image space...")
            image_space_data = data_processing_confidence.convert_kspace_to_image_space(data_path)
            if image_space_data is None:
                print("Failed to convert k-space data to image space.")
                return
            image_space_image_path = data_processing_confidence.save_image_as_png(image_space_data)
            if not image_space_image_path:
                print("Failed to save image space data as PNG.")
                return
            print(f"Image space data saved at: {image_space_image_path}")
            image_url = data_processing_confidence.upload_to_dropbox(image_space_image_path)
            if not image_url:
                print("Failed to upload image space data to Dropbox.")
                return
            print(f"Converted image uploaded to Dropbox. Accessible at: {image_url}")
            processed_image_path = image_space_image_path
        else:
            print("GPT-4o classified the data as image space.")
        final_prompt = (
            f"{prompt}\n"
            f"You are an MRI Agent who can assess MRI data and detect what kind of corruption it has. Analyze the following MRI image and classify the type of corruption it has "
            f"(e.g., motion corrupted, undersampled, noisy, no corruption):\n"
            f"{image_url}\n"
            "You should never omit any of the 5 points below\n"
            "If a confidence score is missing, do not omit it—provide your best estimate."
            "Please output your response in exactly five sections: \n"
            "1. Classification: \n"
            "2. Reasoning: \n"
            "3. Recommended Model: [CycleGAN for MRI Denoising / CycleGAN for MRI Motion Correction / CycleGAN for MRI Reconstruction] \n"
            "4. Correction Plan: \n"
            "5. A confidence score in the exact format: 'Confidence Score: [0-1]: \n"
            "**Important note for response structure**: Do not use newline characters between numbered sections \n"
            "You must include all five sections without merging any two."
        )
        models = [
            ("GPT4o Assistant", self.gpt4o_image_space),
            ("Gemini-2.0 Flash Assistant", self.gemini_corruption)
        ]
        classification_results = []
        confidence_scores = []
        all_responses = []
        for model_name, model_interface in models:
            print(f"\n 🤖 Sending classification request to {model_name}...")
            response = model_interface.get_plan(image_url, final_prompt)
            if not response:
                print(f"Failed to get a response from {model_name}.")
                logging.warning(f"{model_name} did not return a response.")
                continue
            print(f"\n {model_name} Response:📝 \n{response}")
            try:
                parsed_response = data_processing_confidence.parse_gpt4o_response(response)
                classification = parsed_response.get("classification", "unknown")
                confidence = float(parsed_response.get("confidence_score", 0.5))
                reasoning = data_processing_confidence.extract_reasoning(response)
                classification_results.append({
                    "model": model_name,
                    "classification": classification,
                    "confidence": confidence,
                    "reasoning": reasoning
                })
                all_responses.append(response)
            except Exception as e:
                print(f"Error parsing response from {model_name}: {e}")
                logging.error(f"Error parsing response from {model_name}: {e}")
                continue
        if len(classification_results) == 0:
            print("⚠️ No valid classifications received from AI models. Falling back to default classification.")
            if confidence_scores:
                highest_confidence_index = confidence_scores.index(max(confidence_scores))
                fallback_classification = {
                    "model": models[highest_confidence_index][0],
                    "classification": "unknown",
                    "confidence": confidence_scores[highest_confidence_index],
                    "reasoning": "Fallback decision due to missing AI model responses."
                }
                classification_results.append(fallback_classification)
            else:
                print("❌ No AI responses available. Cannot proceed with classification.")
                return
        print(f"\n Performing few-shot learning by Radiologist........")
        gpt4o_few_shot = few_shot_gpt4o.GPT4oFewShot()
        radiologist_response = gpt4o_few_shot.get_few_shot_response(image_url,
                                                                    classification_results)
        if radiologist_response:
            print(f"\n 🧑‍⚕️ Radiologist's Few-Shot Learning Response: 📝 \n{radiologist_response}")
        else:
            print("❌ Radiologist Few-Shot Response is empty.")
        radiologist_decision = data_processing_confidence.parse_evaluator_response(radiologist_response, evaluator_type="radiologist")
        principal_evaluator_prompt_1 = f"""
        You are the Principal Investigator (Senior MRI Expert).
        Your first task is to **independently classify** the corruption in this MRI scan without considering any other responses.

        ### Important Guidelines:
        - Analyze the MRI image independently without any bias from other evaluations.
        - Do NOT assume previous AI or Radiologist classifications are correct.
        - Your classification must be based only on the MRI image itself.
        - You must explain your reasoning clearly based on MRI artifact characteristics.
        - Do NOT include any information about the assistants or radiologist.

        ### Independent Classification
        - Final Classification: [Motion Corrupted / Undersampled / Noisy / Other]
        - Confidence Score (0-1): [Your confidence level]
        - Reasoning: [Explain the classification based only on the image]
        - Final Recommended Correction Model:** [CycleGAN for MRI Denoising / CycleGAN for MRI Motion Correction / CycleGAN for MRI Reconstruction]
        """
        print("\n🔍 Sending **Independent** Classification Request to Principal Investigator...")
        pi_initial_response = self.principal_evaluator.get_agent_plan(image_url, principal_evaluator_prompt_1)
        pi_initial_decision = data_processing_confidence.parse_evaluator_response(pi_initial_response,
                                                                                  evaluator_type="principal")
        print(f"\n👨‍⚕️ **Principal Investigator's Independent Classification**: \n{pi_initial_response}")
        principal_evaluator_prompt = f"""
        Now that you have independently classified the MRI corruption, review the AI assistants' and radiologist's classifications.

        ### **Your Previous Independent Classification:**
        - **Final Classification:** {pi_initial_decision.get("final_classification", "unknown")}
        - **Confidence Score:** {pi_initial_decision.get("recommended_model", "unknown")}
        - **Reasoning:** {pi_initial_decision.get("reasoning", "No justification provided.")}

        ### **Step 2: Review the Assistants’ Classifications**
        Below are the AI assistant responses:

        - **Assistant 1 (GPT-4o Image Space)**:
          - Classification: {classification_results[0]["classification"]}
          - Confidence: {classification_results[0]["confidence"]}
          - Justification: {classification_results[0]["reasoning"]}

        - **Assistant 2 (Gemini-1.5 Flash)**:
          - Classification: {classification_results[1]["classification"]}
          - Confidence: {classification_results[1]["confidence"]}
          - Justification: {classification_results[1]["reasoning"]}

        Evaluate:
        - Do their responses align with MRI physics and your own classification?
        - Did either assistant show a high confidence in a wrong classification?
        - Are their justifications valid?

        ### **Step 3: Review the Radiologist’s Evaluation**
        - **Evaluated Classification:** {radiologist_decision.get("classification", "unknown")}
        - **Confidence Score:** {radiologist_decision.get("confidence_score", 0.5)}
        - **Justification:** {radiologist_decision.get("reasoning", "No justification provided.")}
        - **Recommended Model:** {radiologist_decision.get("recommended_model", "unknown")}

        Evaluate:
        - Did the Radiologist provide a strong justification?
        - If they agreed with the assistants, was their reasoning solid?
        - Did they miss a key artifact in the MRI image?

        ### **Step 4: Make Your Final Decision**
        **IMPORTANT:**
        1. Your classification should remain **{pi_initial_decision.get("final_classification", "unknown")}** unless the assistants and radiologist provide **strong, undeniable evidence** that your classification was incorrect.
        2. If you choose to change your classification, provide a **clear and explicit reason** why your initial classification was wrong.
        3. **Remember Among the other 2 agents, radiologist and Your classification, your classification is the strongest. Because you were fine-tuned on this. So don't get biased by their classification**

        - What did you previously Classify?
        - Are you changing your Decision on Classification? Why? provide proper reasoning:
        - Final Classification: [Confirm or update your previous classification]
        - Agreement with Assistants Responses: [Agree with both / One / None]
        - Agreement with Radiologist Evaluation: [Yes/No]
        - Final Confidence Score: [Update confidence if necessary]
        - Final Recommended Correction Model: [CycleGAN for MRI Denoising / CycleGAN for MRI Motion Correction / CycleGAN for MRI Reconstruction]
        - **Final Justification:** [Explain why you kept or changed your classification]
        """
        print("\n Sending evaluation request to 👨‍⚕️ Principal Investigator...")
        principal_response = self.principal_evaluator.get_agent_plan(image_url, principal_evaluator_prompt)
        principal_decision = data_processing_confidence.parse_evaluator_response(principal_response, evaluator_type="principal")
        print(f"\n 👨‍⚕️ Principal Agent's Response: 📝 \n{principal_response}")
        print("\n--- 🧑‍⚕️ Radiologist Evaluation ---")
        print(f"Classification: {radiologist_decision.get('classification', 'Unknown')}")
        print(f"Confidence: {radiologist_decision.get('confidence_score', 'N/A')}")
        print(f"Reasoning: {radiologist_decision.get('reasoning', 'No justification provided.')}")
        print(f"Recommended Model: {radiologist_decision.get('recommended_model', 'Unknown')}\n")
        print("\n--- 👨‍⚕️ Principal Investigator Evaluation ---")
        print(f"Final Classification: {principal_decision.get('final_classification', 'Unknown')}")
        print(f"Agreement with Assistants: {principal_decision.get('agreement_with_assistants', 'Unknown')}")
        print(f"Agreement with Radiologist: {principal_decision.get('agreement_with_radiologist', 'Unknown')}")
        print(f"Confidence: {principal_decision.get('confidence_score', 'N/A')}")
        print(f"Final Recommended Model: {principal_decision.get('recommended_model', 'Unknown')}")
        print(f"Final Justification: {principal_decision.get('reasoning', 'No justification provided.')}\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta_model = MetaModel(input_dim=15, hidden_dim=64, output_dim=3).to(device)
        meta_model.load_state_dict(torch.load(r"D:\MRI-AgentNet\model_selection/meta_model_best.pth", map_location=device))
        meta_model.eval()
        correction_model_mapping = {
            0: "motion_correction",
            1: "denoising",
            2: "reconstruction"
        }
        print("\n⚙️ Running MetaModel-Based Model Selection...\n")
        def map_classification(classification):
            classification = classification.lower().strip()
            classification_mappings = {
                "k-space artifact": "undersampled",
                "k-space artifact or undersampling": "undersampled",
                "motion artifact": "motion corrupted",
                "motion corrupted": "motion corrupted",
                "noisy": "noisy",
                "noisy or noise corrupted": "noisy",
                "radiofrequency noise": "noisy",
                "gradient instability": "noisy",
                "mri noise": "noisy",
                "unknown": "noisy"
            }
            return classification_mappings.get(classification, classification)
        def one_hot_encode(classification, corruption_types=["undersampled", "motion corrupted", "noisy"]):
            one_hot = [0] * len(corruption_types)
            mapped_classification = map_classification(classification)
            if mapped_classification in corruption_types:
                one_hot[corruption_types.index(mapped_classification)] = 1
            return one_hot
        print("\n🔍 **Pre-Encoding Debugging**:")
        print(f"PI Final Classification: {pi_initial_decision.get('final_classification', 'unknown')}")
        print(
            f"Standardized Classification: {map_classification(pi_initial_decision.get('final_classification', 'unknown'))}")
        print(
            f"Assistant 1 Classification: {classification_results[0]['classification']} → {map_classification(classification_results[0]['classification'])}")
        print(
            f"Assistant 2 Classification: {classification_results[1]['classification']} → {map_classification(classification_results[1]['classification'])}")
        print(
            f"Radiologist Classification: {radiologist_decision.get('classification', 'unknown')} → {map_classification(radiologist_decision.get('classification', 'unknown'))}")
        input_vector = (
                one_hot_encode(map_classification(
                    pi_initial_decision.get("final_classification", "unknown"))) +
                one_hot_encode(
                    map_classification(classification_results[0]["classification"])) +
                one_hot_encode(
                    map_classification(classification_results[1]["classification"])) +
                one_hot_encode(map_classification(
                    radiologist_decision.get("classification", "unknown"))) +
                one_hot_encode(map_classification(pi_initial_decision.get("final_classification", "unknown")))
        )
        print(f"✅ **Final Encoded Input Vector for MetaModel:** {input_vector} (Expected size: 15)")
        print(f"Encoded Input Vector for MetaModel: {input_vector}")
        input_tensor = torch.tensor([input_vector], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = meta_model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        print(f"\n🔍 **MetaModel Raw Output Vector:** {output.cpu().numpy()}")
        print(f"Predicted Label: {predicted_label} → {correction_model_mapping[predicted_label]}")
        selected_model = correction_model_mapping[predicted_label]
        print(f"🤝🔬 Multi-Agent Final Decision: {principal_decision.get('final_classification', 'Unknown')} "
              f"(Confidence: {principal_decision.get('confidence_score', 'N/A')})")
        print(f"✅ Final Correction Model Recommendation (by MetaModel): {selected_model}")
        gpt4o_decision = data_processing_confidence.parse_evaluator_response(all_responses[0],
                                                                             evaluator_type="assistant")
        gemini_decision = data_processing_confidence.parse_evaluator_response(all_responses[1],
                                                                              evaluator_type="assistant")
        radiologist_decision = data_processing_confidence.parse_evaluator_response(radiologist_response,
                                                                                   evaluator_type="radiologist")
        principal_decision = data_processing_confidence.parse_evaluator_response(principal_response,
                                                                                 evaluator_type="principal")
        if selected_model != 'no_corruption':
            self.run_model(selected_model, processed_image_path, self.results_dir, self.checkpoints_dir)
        else:
            print("✅ No correction model needed based on final decision ('no_corruption').")
    def _setup_temp_input(self, data_path):
            try:
                temp_dir = tempfile.mkdtemp()
                input_dir = os.path.join(temp_dir, 'input_images')
                os.makedirs(input_dir, exist_ok=True)
                unique_filename_stem = f"{uuid.uuid4().hex}_{os.path.splitext(os.path.basename(data_path))[0]}"
                file_extension = os.path.splitext(data_path)[1]
                unique_filename = f"{unique_filename_stem}{file_extension}"
                input_image_path = os.path.join(input_dir, unique_filename)
                shutil.copy(data_path, input_image_path)
                print(f"   Copied input image to temporary path: {input_image_path}")
                return temp_dir, input_dir, input_image_path, unique_filename_stem
            except Exception as e:
                print(f"❌ Error setting up temporary input directory: {e}")
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise
    def _setup_output_paths(self, results_dir_base, model_type):
            model_result_folder_mapping = {
                "motion_correction": "severe_motion",
                "denoising": "denoising_model",
                "reconstruction": "newly_undersampled"
            }
            result_folder_name = model_result_folder_mapping.get(model_type, model_type)
            output_dir_root = os.path.join(results_dir_base, result_folder_name, 'test_latest')
            output_dir_images = os.path.join(output_dir_root, 'images')
            try:
                os.makedirs(output_dir_images, exist_ok=True)
                print(f"   Ensured output image directory exists: {output_dir_images}")
                return output_dir_images, output_dir_root
            except OSError as e:
                print(f"❌ Error creating output directories: {e}")
                raise
    def _cleanup_previous_outputs(self, output_dir_images, output_dir_root, suffix="_fake.png"):
            count = 0
            for directory in [output_dir_images, output_dir_root]:
                if os.path.exists(directory):
                    try:
                        for filename in os.listdir(directory):
                            if filename.lower().endswith(suffix):
                                file_path = os.path.join(directory, filename)
                                try:
                                    os.remove(file_path)
                                    count += 1
                                except OSError as e:
                                    print(f"⚠️ Could not remove old file {file_path}: {e}")
                    except OSError as e:
                        print(f"⚠️ Could not list files in {directory} for cleanup: {e}")
            if count > 0:
                print(f"🧹 Removed {count} old '{suffix}' images from output directories.")
    def _prepare_inference_options(self, input_dir, model_type, results_dir_base, checkpoints_dir_base):
            args = [
                '--dataroot', input_dir,
                '--model', 'test',
                '--no_dropout',
                '--serial_batches',
                '--eval',
                '--num_test', '1',
                '--phase', 'test',
                '--direction', 'AtoB',
                '--epoch', 'latest',
                '--dataset_mode', 'single',
                '--load_size', '396',
                '--crop_size', '396',
                '--results_dir', results_dir_base,
                '--gpu_ids', '-1',
                '--no_html',
                '--model_suffix', ''
            ]
            model_checkpoint_mapping = {
                "motion_correction": "severe_motion",
                "denoising": "denoising_model",
                "reconstruction": "newly_undersampled"
            }
            model_name = model_checkpoint_mapping.get(model_type)
            if not model_name:
                print(f"⚠️ Model type '{model_type}' has no specific mapping for 'name'. Using default logic.")
                args.extend(['--name', f"{model_type}_model"])
                args.extend(['--checkpoints_dir', checkpoints_dir_base])
            else:
                args.extend(['--name', model_name])
                args.extend(['--checkpoints_dir', checkpoints_dir_base])
            print(f"   Parsing options for model: {model_name or model_type}")
            try:
                opt = TestOptions().parse(args)
            except Exception as e:
                print(f"❌ Failed to parse TestOptions with args: {args}. Error: {e}")
                raise
            opt.num_threads = 0
            opt.batch_size = 1
            opt.isTrain = False
            print(
                f"   Inference options prepared: name={opt.name}, results={opt.results_dir}, checkpoints={opt.checkpoints_dir}")
            return opt
    def _find_output_image_with_polling(self, output_dir, expected_suffix="_fake.png", timeout=120,
                                            poll_interval=0.5):
            print(f"   Waiting for output file ending in '{expected_suffix}' in {output_dir} (timeout: {timeout}s)...")
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if not os.path.exists(output_dir):
                        pass
                    else:
                        for filename in os.listdir(output_dir):
                            if filename.lower().endswith(expected_suffix):
                                output_image_path = os.path.join(output_dir, filename)
                                print(f"   Found output file: {output_image_path}")
                                return output_image_path
                except OSError as e:
                    print(f"⚠️ Error listing files in {output_dir} while polling: {e}")
                time.sleep(poll_interval)
            print(f"❌ Timeout: Output file with suffix '{expected_suffix}' not found after {timeout} seconds.")
            return None
    def _display_results(self, input_path, output_path):
            global _DISPLAY_AVAILABLE
            if not _DISPLAY_AVAILABLE:
                print("   Skipping image display because matplotlib/cv2 are not available.")
                return
            try:
                import matplotlib.pyplot as plt
                import cv2
            except ImportError:
                print("   Error: Failed to import matplotlib or cv2 for display.")
                return
            if not input_path or not os.path.exists(input_path):
                print(f"❌ Cannot display results: Input image missing at {input_path}")
                return
            if not output_path or not os.path.exists(output_path):
                print(f"❌ Cannot display results: Output image missing at {output_path}")
                return
            try:
                input_img = cv2.imread(input_path)
                output_img = cv2.imread(output_path)
                if input_img is None:
                    print(f"❌ Failed to load input image from {input_path}")
                    return
                if output_img is None:
                    print(f"❌ Failed to load output image from {output_path}")
                    return
                input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(input_img_rgb)
                axes[0].set_title("🛑 Input Image (Corrupted/Original)")
                axes[0].axis("off")
                axes[1].imshow(output_img_rgb)
                axes[1].set_title("✅ Corrected Output Image")
                axes[1].axis("off")
                plt.tight_layout()
                print("   Displaying input and output images...")
                plt.show()
            except Exception as e:
                print(f"❌ Error displaying images: {e}")
    def run_model(self, model_type, data_path, results_dir_base, checkpoints_dir_base):
            print(f"\n⚙️ Running correction model: type='{model_type}', input='{data_path}'...")
            temp_dir = None
            final_output_image_path = None
            try:
                temp_dir, input_dir, temp_input_image_path, unique_filename_stem = self._setup_temp_input(data_path)
                output_dir_images, output_dir_root = self._setup_output_paths(results_dir_base, model_type)
                print(f"   Expecting output images in: {output_dir_images}")
                self._cleanup_previous_outputs(output_dir_images, output_dir_root, suffix="_fake.png")
                opt = self._prepare_inference_options(input_dir, model_type, results_dir_base, checkpoints_dir_base)
                print(f"   Starting inference process for model '{opt.name}'...")
                from models.test import run_inference
                run_inference(opt)
                print("   Inference function call completed.")
                print(f"   Polling for output file in directory: {output_dir_root}")
                raw_output_image_path = self._find_output_image_with_polling(
                    output_dir_root,
                    expected_suffix="_fake.png",
                    timeout=600
                )
                if raw_output_image_path:
                    final_output_image_path = os.path.join(output_dir_images, f"{unique_filename_stem}_fake.png")
                    print(f"   Preparing final output path: {final_output_image_path}")
                    os.makedirs(os.path.dirname(final_output_image_path), exist_ok=True)
                    if os.path.abspath(raw_output_image_path) != os.path.abspath(final_output_image_path):
                        if os.path.exists(final_output_image_path):
                            print(f"⚠️ Removing existing file at final destination: {final_output_image_path}")
                            try:
                                os.remove(final_output_image_path)
                            except OSError as e:
                                print(f"❌ Could not remove existing file {final_output_image_path}: {e}")
                        try:
                            print(
                                f"   Moving {raw_output_image_path} \n     to --> {final_output_image_path}")
                            shutil.move(raw_output_image_path, final_output_image_path)
                            print(f"✅ Output image successfully saved at: {final_output_image_path}")
                        except Exception as e:
                            print(f"❌ Failed to move output file: {e}")
                            final_output_image_path = None
                    else:
                        print(
                            f"   Output file already has the final desired name and location: {raw_output_image_path}")
                        final_output_image_path = raw_output_image_path
                else:
                    print(
                        f"❌ Polling failed: Could not find the generated output image ending with '_fake.png' in {output_dir_root}.")
                    final_output_image_path = None
            except Exception as e:
                print(f"❌ An error occurred during model execution: {e}")
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"🧹 Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        print(f"❌ Error cleaning up temporary directory {temp_dir}: {e}")
            if final_output_image_path:
                self._display_results(data_path, final_output_image_path)
            else:
                print("   Skipping display as no final output image path was determined.")
            print(f"🏁 Correction model run finished for type='{model_type}'.")
if __name__ == "__main__":
        start_time = time.time()
        client = OpenAI(api_key="API_KEY")
        root = Tk()
        root.withdraw()
        data_path = askopenfilename(title="Select the MRI data file",
                                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg"),
                                               ("MAT files", "*.mat"),
                                               ("HDF5 files", "*.hdf5;*.h5"),
                                               ("Numpy files", "*.npy"),
                                               ("NIfTI files", "*.nii;*.nii.gz"),
                                               ("DICOM files", "*.dcm")])
        if not data_path:
            print("No file selected. Exiting.")
            sys.exit()
        else:
            print(f"Selected data file: {data_path}")
        prompt = input("Please provide the prompt (e.g., 'Analyze this MRI scan'): ")
        if not prompt:
            prompt = "Analyze this MRI scan for corruptions and suggest corrections."
            print(f"Using default prompt: {prompt}")
        try:
            agent = MRI_Agent(client=client)
            agent.process(data_path, prompt)
        except Exception as e:
            print(f"\n❌ An critical error occurred in the main execution block: {e}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal Execution Time: {execution_time:.2f} seconds")

