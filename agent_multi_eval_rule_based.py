import sys
import os
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
from gemini_interface_confidence import GeminiInterface
from utils.bert_score_utils import compute_bert_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpt4o_interface import GPT4oInterface
from plan_parser import PlanParser
from model_selector import ModelSelector
from openai import OpenAI
from utils import data_processing_confidence
from models.options.test_options import TestOptions
from models.test import run_inference
class MRI_Agent:
    def __init__(self, client, model_image_space="gpt-4o-2024-11-20", model_corruption1="o1-2024-12-17", model_principal="ft:gpt-4o-2024-08-06:personal:combined-900:Av7pB23x"):
        self.gpt4o_image_space = GPT4oInterface(client=client, model=model_image_space)
        self.gpt4o_corruption = GPT4oInterface(client=client, model=model_corruption1)
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.gemini_corruption = GeminiInterface(self.gemini_model)
        self.gemini_model2 = genai.GenerativeModel(model_name="gemini-2.0-flash")
        self.radiologist_evaluator = GeminiInterface(self.gemini_model2)
        self.principal_evaluator = GPT4oInterface(client=client, model=model_principal)
        self.parser = PlanParser()
        self.model_selector = ModelSelector()
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
            "You must include the following details in **every response**:\n"
            "1. Classification of the corruption.\n"
            "2. Reasoning behind your classification. \n"
            "3. Recommended model to address the corruption.\n"
            "4. A detailed plan to correct the corruption.\n"
            "5. A confidence score in the exact format: 'Confidence Score: [0-1]'.\n"
            "If a confidence score is missing, do not omit it—provide your best estimate."
        )
        models = [
            ("GPT-4o Image Space", self.gpt4o_image_space),
            ("Gemini-1.5 Flash", self.gemini_corruption)
        ]
        classification_results = []
        confidence_scores = []
        all_responses = []
        for model_name, model_interface in models:
            print(f"Sending classification request to {model_name}...")
            response = model_interface.get_plan(image_url, final_prompt)
            if not response:
                print(f"Failed to get a response from {model_name}.")
                logging.warning(f"{model_name} did not return a response.")
                continue
            print(f"{model_name} Response:\n{response}")
            try:
                parsed_response = data_processing_confidence.parse_gpt4o_response(response)
                classification = parsed_response.get("classification", "unknown")
                confidence = float(parsed_response.get("confidence_score", 0.5))
                reasoning = parsed_response.get("reasoning", "No reasoning provided.")
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
        junior_evaluator_prompt = f"""
        Two AI models have classified this MRI corruption. You are a Radiologist. Your task is to assess the correctness of their classifications.

        ### Model Classifications:
        1️⃣ {classification_results[0]["model"]}: {classification_results[0]["classification"]}
           Confidence: {classification_results[0]["confidence"]}
           Justification: {classification_results[0]["reasoning"]}

        2️⃣ {classification_results[1]["model"]}: {classification_results[1]["classification"]}
           Confidence: {classification_results[1]["confidence"]}
           Justification: {classification_results[1]["reasoning"]}

        ### Output Format:
        - Evaluated Classification: [Classification decision]
        - What the two assistants said about the corruption: [Assistants Classification]
        - Agree with both Assistants? : [Agree or Disagree]
        - Confidence Score: [0-1]
        - Justification: [Detailed reasoning]
        - Recommended Model: [CycleGAN for MRI denoising/ CycleGAN for MRI motion Correction / CycleGAN for MRI Reconstruction]
        """
        print("Sending evaluation request to Radiologist Agent...")
        radiologist_response = self.radiologist_evaluator.get_agent_plan(image_url, junior_evaluator_prompt)
        radiologist_decision = data_processing_confidence.parse_evaluator_response(radiologist_response, evaluator_type="radiologist")
        print(f"\n 🩺 Junior Agent's Response:\n{radiologist_response}")
        principal_evaluator_prompt = f"""
        You are the Principal Investigator (Senior MRI Expert) who can detect different MRI corruption very accurately, whether it's motion/noise/undersampling. Your task is to review the following evaluations and make a final decision.
        The assistants and radiologist may provide wrong classification, you can't blindly accept their classification. Sometimes assistants and radiologist may misclassify
        undersampled as motion corrupted. You have to be careful about that.

        ### Assistant Model Responses:
        - **Assistant 1 (GPT-4o Image Space):**
          - Classification: {classification_results[0]["classification"]}
          - Confidence: {classification_results[0]["confidence"]}
          - Justification: {classification_results[0]["reasoning"]}

        - **Assistant 2 (Gemini-1.5 Flash):**
          - Classification: {classification_results[1]["classification"]}
          - Confidence: {classification_results[1]["confidence"]}
          - Justification: {classification_results[1]["reasoning"]}

        ### Radiologist Evaluation:
        - Evaluated Classification: {radiologist_decision.get("classification", "unknown")}
        - Confidence Score: {radiologist_decision.get("confidence_score", 0.5)}
        - Justification: {radiologist_decision.get("reasoning", "No justification provided.")}
        - Recommended Model: {radiologist_decision.get("recommended_model", "unknown")}

        ### Your Task:
        1. Do both the assistants responses align with expert MRI knowledge?
        2. Is the Radiologist's evaluation justified?
        3. Considering both the assistant responses and the Radiologist's evaluation, provide your final decision.

        **Output Format:**
        - Final Classification: [Final MRI corruption classification]
        - Agreement with Assistants Responses: [Agree with both / One / None]
        - Agreement with Radiologist Evaluation: [Yes/No]
        - Final Confidence Score: [0-1]
        - Final Recommended Correction Model: [CycleGAN for MRI denoising/ CycleGAN for MRI motion Correction / CycleGAN for MRI Reconstruction]
        - Final Justification: [Detailed reasoning why agree or disagree]
        """
        print("Sending evaluation request to Principal Investigator...")
        principal_response = self.principal_evaluator.get_agent_plan(image_url, principal_evaluator_prompt)
        principal_decision = data_processing_confidence.parse_evaluator_response(principal_response, evaluator_type="principal")
        print(f"\n 👨‍⚕️ Principal Agent's Response:\n{principal_response}")
        print("\n--- 🩺 Radiologist Evaluation ---")
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
        pi_final_classification = principal_decision.get("final_classification", "").strip().lower()
        pi_agreement = principal_decision.get("agreement", "").strip().lower()
        if (pi_final_classification and pi_final_classification != "unknown") or (pi_agreement == "no"):
            final_classification = principal_decision.get("final_classification",
                                                          radiologist_decision.get("classification", "unknown"))
            final_model = principal_decision.get("recommended_model",
                                                 radiologist_decision.get("recommended_model", "unknown"))
            final_confidence = principal_decision.get("confidence_score",
                                                      radiologist_decision.get("confidence_score", 0.5))
        else:
            final_classification = radiologist_decision.get("classification", "unknown")
            final_model = radiologist_decision.get("recommended_model", "unknown")
            final_confidence = radiologist_decision.get("confidence_score", 0.5)
        print(f"🧠 Multi-Agent Final Decision: {final_classification} (Confidence: {final_confidence:.2f})")
        print(f"✅ Final Correction Model Recommendation: {final_model}")
        assistant_justifications = [
            classification_results[0]["reasoning"],
            classification_results[1]["reasoning"]
        ]
        radiologist_justification = radiologist_decision.get("reasoning", "No justification provided.")
        principal_justification = principal_decision.get("reasoning", "No justification provided.")
        assistant_vs_radiologist_score = compute_bert_score(
            reference_texts=[radiologist_justification] * 2,
            candidate_texts=assistant_justifications
        )
        radiologist_vs_principal_score = compute_bert_score(
            reference_texts=[radiologist_justification],
            candidate_texts=[principal_justification]
        )
        assistant_vs_principal_score = compute_bert_score(
            reference_texts=[principal_justification] * 2,
            candidate_texts=assistant_justifications
        )
        print("\n--- 🔍 Debugging BERTScore Inputs ---")
        print(f"📝 Assistant Justifications: {assistant_justifications}")
        print(f"📝 Radiologist Justification: {radiologist_justification}")
        print(f"📝 Principal Investigator Justification: {principal_justification}")
        print(f"🔹 **Assistant (GPT-4o & Gemini) vs. Radiologist BERTScore**: "
              f"{sum(assistant_vs_radiologist_score) / len(assistant_vs_radiologist_score):.4f}")
        print(f"🔹 **Radiologist vs. Principal Investigator BERTScore**: {radiologist_vs_principal_score[0]:.4f}")
        print(f"🔹 **Assistant (GPT-4o & Gemini) vs. Principal Investigator BERTScore**: "
              f"{sum(assistant_vs_principal_score) / len(assistant_vs_principal_score):.4f}")
        if sum(assistant_vs_radiologist_score) / len(assistant_vs_radiologist_score) < 0.5:
            print("⚠️ Warning: Radiologist's assessment deviates significantly from AI assistants.")
        if radiologist_vs_principal_score[0] < 0.6:
            print("⚠️ Warning: Principal Investigator disagrees significantly with the Radiologist.")
        if sum(assistant_vs_principal_score) / len(assistant_vs_principal_score) < 0.5:
            print("⚠️ Warning: Principal Investigator disagrees significantly with AI assistants.")
        plan_details = self.parser.parse_plan(all_responses[0])
        model_type = self.model_selector.select_model(plan_details)
        if model_type:
            print(f"\nSelected Model: {model_type}")
            self.run_model(model_type, processed_image_path)
        else:
            print("No suitable model found for the given classification.")
    def run_model(self, model_type, data_path):
        print(f"\nRunning {model_type} model on the provided MRI data from {data_path}...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        sys.path.insert(0, project_root)
        from models.options.test_options import TestOptions
        from models.test import run_inference
        from utils.data_processing_confidence import read_mri_data, combine_coils, save_image_as_png
        temp_dir = tempfile.mkdtemp()
        try:
            input_dir = os.path.join(temp_dir, 'input_images')
            os.makedirs(input_dir, exist_ok=True)
            unique_filename = f"{uuid.uuid4().hex}_{os.path.basename(data_path)}"
            input_image_path = os.path.join(input_dir, unique_filename)
            shutil.copy(data_path, input_image_path)
            print(f"Copied input image to {input_image_path}")
            results_dir = r'C:\Users\gsaju\OneDrive - University of Massachusetts Dartmouth\Documents\CVPR 2024\AgentMRI\models\results'
            output_dir_images = os.path.join(results_dir, f"{model_type}_model", 'test_latest', 'images')
            output_dir_root = os.path.join(results_dir, f"{model_type}_model", 'test_latest')
            os.makedirs(output_dir_images, exist_ok=True)
            for directory in [output_dir_images, output_dir_root]:
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        if file.lower().endswith("_fake.png"):
                            os.remove(os.path.join(directory, file))
                            print(f"Removed existing output file: {file} in {directory}")
            args = [
                '--dataroot', input_dir,
                '--name', f"{model_type}_model",
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
                '--results_dir', results_dir,
                '--gpu_ids', '-1',
                '--no_html',
                '--model_suffix', ''
            ]
            opt = TestOptions().parse(args)
            if model_type == 'motion_correction':
                opt.name = 'severe_motion'
                opt.checkpoints_dir = r'C:\Users\gsaju\OneDrive - University of Massachusetts Dartmouth\Documents\CVPR 2024\AgentMRI\models\checkpoints'
            elif model_type == 'denoising':
                opt.name = 'denoising_model'
                opt.checkpoints_dir = r'C:\Users\gsaju\OneDrive - University of Massachusetts Dartmouth\Documents\CVPR 2024\AgentMRI\models\checkpoints'
            elif model_type == 'reconstruction':
                opt.name = 'newly_undersampled'
                opt.checkpoints_dir = r'C:\Users\gsaju\OneDrive - University of Massachusetts Dartmouth\Documents\CVPR 2024\AgentMRI\models\checkpoints'
            elif model_type == 'no_corruption':
                print(f"No correction required, Image is Corruption free")
            else:
                print(f"Model type '{model_type}' not recognized.")
                return
            opt.num_threads = 0
            opt.batch_size = 1
            opt.isTrain = False
            run_inference(opt)
            time.sleep(2)
            print(f"Listing files in output_dir_images ({output_dir_images}):")
            try:
                files_in_images = os.listdir(output_dir_images)
                for f in files_in_images:
                    print(f" - {f}")
            except FileNotFoundError:
                print(f"Directory {output_dir_images} does not exist.")
            print(f"Listing files in output_dir_root ({output_dir_root}):")
            try:
                files_in_root = os.listdir(output_dir_root)
                for f in files_in_root:
                    print(f" - {f}")
            except FileNotFoundError:
                print(f"Directory {output_dir_root} does not exist.")
            combined_output = []
            if os.path.exists(output_dir_images):
                combined_output += [os.path.join(output_dir_images, f) for f in files_in_images if f.lower().endswith(('.png', '.jpg'))]
            if os.path.exists(output_dir_root):
                combined_output += [os.path.join(output_dir_root, f) for f in files_in_root if f.lower().endswith(('.png', '.jpg'))]
            print(f"Found {len(combined_output)} output images in combined directories.")
            if combined_output:
                output_image_path = combined_output[0]
                print(f"First output image found: {output_image_path}")
                final_output_image_path = os.path.join(output_dir_images, f"{unique_filename}_fake.png")
                if os.path.exists(final_output_image_path):
                    os.remove(final_output_image_path)
                    print(f"Removed existing final output image: {final_output_image_path}")
                shutil.move(output_image_path, final_output_image_path)
                print(f"Output image saved at {final_output_image_path}")
            else:
                print("No output image generated.")
        except Exception as e:
            print(f"Error running the model: {e}")
        finally:
            shutil.rmtree(temp_dir)
        print("Model processing complete!")
if __name__ == "__main__":
    start_time = time.time()
    client = OpenAI(api_key="API_KEY")
    root = Tk()
    root.withdraw()
    data_path = askopenfilename(title="Select the MRI data file",
                                filetypes=[("IMG files", "*.png;*.jpg;*.jpeg"),
                                           ("MAT files", "*.mat"),
                                           ("HDF5 files", "*.hdf5;*.h5"),
                                           ("Numpy files", "*.npy"),
                                           ("NIfTI files", "*.nii;*.nii.gz"),
                                           ("DICOM files", "*.dcm")])
    if not data_path:
        print("No file selected. Exiting.")
        sys.exit()
    prompt = input("Please provide the prompt: ")
    agent = MRI_Agent(client=client)
    agent.process(data_path, prompt)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
