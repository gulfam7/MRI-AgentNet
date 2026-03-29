import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import dropbox
import h5py
import spacy
import scipy.io
import nibabel as nib
import pydicom
from PIL import Image
import re
import torch
from bert_score import score
def read_mri_data(file_path):
    """
    Reads MRI data from the given file path.

    Parameters:
    - file_path: Path to the MRI data file.

    Returns:
    - mri_data: NumPy array containing MRI data or file path if already an image.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.mat':
        mri_data = read_mat_file(file_path)
    elif ext in ['.h5', '.hdf5']:
        mri_data = read_hdf5_file(file_path)
    elif ext == '.npy':
        mri_data = read_npy_file(file_path)
    elif ext in ['.nii', '.nii.gz']:
        mri_data = read_nifti_file(file_path)
    elif ext == '.dcm':
        mri_data = read_dicom_file(file_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        print("Image file detected. No processing required.")
        return file_path                                        
    else:
        print(f"Unsupported file extension: {ext}")
        return None

    return mri_data

def read_mat_file(file_path):
    """
    Reads .mat files and returns MRI data as a NumPy array.
    """
    try:
        mat_contents = scipy.io.loadmat(file_path)

        data_keys = [key for key in mat_contents.keys() if not key.startswith('__')]

        if not data_keys:
            print("No valid data variable found in the .mat file.")
            return None
        elif len(data_keys) == 1:

            variable_name = data_keys[0]
        else:

            variable_name = max(data_keys, key=lambda k: mat_contents[k].size)
            print(f"Automatically selecting the largest variable: '{variable_name}'")

        mri_data = mat_contents[variable_name]
        return mri_data
    except Exception as e:
        print(f"Error reading .mat file: {e}")
        return None

def read_hdf5_file(file_path):
    """
    Reads .hdf5/.h5 files and returns MRI data as a NumPy array.
    """
    try:
        with h5py.File(file_path, 'r') as f:

            def find_datasets(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets[name] = node

            datasets = {}
            f.visititems(find_datasets)

            if not datasets:
                print("No datasets found in the .hdf5 file.")
                return None
            elif len(datasets) == 1:

                dataset_name = list(datasets.keys())[0]
            else:

                dataset_name = max(datasets, key=lambda k: datasets[k].size)
                print(f"Automatically selecting the largest dataset: '{dataset_name}'")

            mri_data = np.array(datasets[dataset_name])
            return mri_data
    except Exception as e:
        print(f"Error reading .hdf5 file: {e}")
        return None

def read_npy_file(file_path):
    """
    Reads .npy files and returns MRI data as a NumPy array.
    """
    try:
        mri_data = np.load(file_path)
        return mri_data
    except Exception as e:
        print(f"Error reading .npy file: {e}")
        return None

def read_nifti_file(file_path):
    """
    Reads NIfTI files (.nii, .nii.gz) and returns MRI data as a NumPy array.
    """
    try:
        nifti_img = nib.load(file_path)
        mri_data = nifti_img.get_fdata()
        return mri_data
    except Exception as e:
        print(f"Error reading NIfTI file: {e}")
        return None

def read_dicom_file(file_path):
    """
    Reads DICOM files and returns MRI data as a NumPy array.
    """
    try:
        dicom_img = pydicom.dcmread(file_path)
        mri_data = dicom_img.pixel_array
        return mri_data
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None

def read_image_file(file_path):
    """
    Reads image files and returns MRI data as a NumPy array.

    Parameters:
    - file_path: Path to the image file.

    Returns:
    - mri_data: NumPy array containing image data.
    """
    try:
        image = Image.open(file_path).convert('L')                        
        mri_data = np.array(image)
        return mri_data
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None

def combine_coils(multi_coil_data):
    """
    Combines multi-coil MRI data using Root Sum of Squares (RSS).

    Parameters:
    - multi_coil_data: NumPy array of shape (height, width, coils)

    Returns:
    - mri_image: Combined MRI image as a 2D NumPy array
    """
    rss_image = np.sqrt(np.sum(np.abs(multi_coil_data) ** 2, axis=-1))
    return rss_image

def save_image_as_png(mri_image):
    """
    Normalizes MRI image data and saves it as a PNG file in RGB format.

    Parameters:
    - mri_image: 2D NumPy array

    Returns:
    - png_path: Path to the saved PNG file
    """
    try:

        if np.iscomplexobj(mri_image):
            print("Complex data detected, converting to magnitude...")
            mri_image = np.abs(mri_image).astype(np.float32)                     


        mri_image_normalized = mri_image - np.min(mri_image)
        if np.max(mri_image_normalized) != 0:
            mri_image_normalized /= np.max(mri_image_normalized)
        mri_image_normalized *= 255.0
        mri_image_normalized = mri_image_normalized.astype(np.uint8)


        image = Image.fromarray(mri_image_normalized).convert('RGB')


        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name)
            png_path = tmp_file.name

        return png_path
    except Exception as e:
        print(f"Error saving image as PNG: {e}")
        return None

def upload_to_dropbox(file_path):
    """
    Uploads a file to Dropbox and returns a shareable, publicly accessible URL.

    Parameters:
    - file_path: Local path to the file to upload

    Returns:
    - url: Publicly accessible URL of the uploaded file
    """
    try:

        DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
        if not DROPBOX_ACCESS_TOKEN:
            print("Dropbox access token not found. Please set the 'DROPBOX_ACCESS_TOKEN' variable.")
            return None


        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


        dropbox_path = f"/AgentMRI/{os.path.basename(file_path)}"


        with open(file_path, 'rb') as f:
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)


        shared_link_metadata = dbx.sharing_create_shared_link_with_settings(
            dropbox_path,
            settings=dropbox.sharing.SharedLinkSettings(requested_visibility=dropbox.sharing.RequestedVisibility.public)
        )


        url = shared_link_metadata.url.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '')

        return url
    except Exception as e:
        print(f"Error uploading to Dropbox: {e}")
        return None

def preprocess_data(data_path):
    """
    Preprocesses MRI data and returns the path to the preprocessed image file.

    Parameters:
    - data_path: Local path to the MRI data file.

    Returns:
    - image_path: Path to the preprocessed image file or the original image path if already in image format.
    """
    if data_path.endswith(('.png', '.jpg', '.jpeg')):
        print("File is already an image. Skipping processing.")
        return data_path

    mri_data = read_mri_data(data_path)
    if mri_data is None:
        print("Failed to read MRI data.")
        return None


    if len(mri_data.shape) == 3:
        print("Multi-coil data detected. Combining coils...")
        mri_data = combine_coils(mri_data)
    elif len(mri_data.shape) == 2:
        print("Single-coil data detected.")
    else:
        print("Unsupported data shape.")
        return None


    image_path = save_image_as_png(mri_data)
    if not image_path:
        print("Failed to save image as PNG.")
        return None

    return image_path

def convert_kspace_to_image_space(data_path):
    """
    Reads the original data and converts k-space data to image space.

    Parameters:
    - data_path: Path to the MRI data file.

    Returns:
    - image_data: NumPy array containing image space data.
    """

    mri_data = read_mri_data(data_path)
    if mri_data is None:
        print("Failed to read MRI data.")
        return None


    if not np.iscomplexobj(mri_data):
        print("Data is not complex-valued. Cannot perform k-space to image space conversion.")
        return None


    image_data = kspace_to_image(mri_data)


    if len(image_data.shape) == 3:
        print("Multi-coil image data detected after k-space conversion. Combining coils...")
        image_data = combine_coils(image_data)
    elif len(image_data.shape) == 2:
        print("Single-coil image data obtained from k-space.")
    else:
        print("Unsupported data shape after k-space conversion.")
        return None

    return image_data

def kspace_to_image(kspace_data):
    """
    Converts k-space data to image space using inverse Fourier Transform.
    """
    if kspace_data.ndim == 3:                   
        image_data = np.zeros_like(kspace_data, dtype=np.float32)
        for i in range(kspace_data.shape[2]):
            coil_kspace = np.fft.ifftshift(kspace_data[:, :, i])
            coil_image = np.fft.ifft2(coil_kspace)
            coil_image = np.fft.fftshift(coil_image)
            image_data[:, :, i] = np.abs(coil_image)
        image_data = np.sqrt(np.sum(image_data ** 2, axis=-1))                          
    else:                    
        kspace_data = np.fft.ifftshift(kspace_data)
        image_data = np.fft.ifft2(kspace_data)
        image_data = np.fft.fftshift(image_data)
        image_data = np.abs(image_data)

    image_data /= np.max(image_data)             
    return image_data

def load_image(image_path):
    """
    Loads an image file and returns it as a NumPy array.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - image: 2D NumPy array of the image.
    """
    try:
        image = Image.open(image_path).convert('L')                        
        image = np.array(image)
        return image
    except Exception as e:
        print(f"Error loading image file: {e}")
        return None


nlp = spacy.load("en_core_web_sm")


def preprocess_response(response):
    """
    Removes Markdown formatting (e.g., asterisks) from the response string.
    """

    clean_response = re.sub(r'[*_`~]', '', response)
    return clean_response


try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except ImportError:

    nlp = None


def parse_gpt4o_response(response):
    """
    Parse GPT-4o response to extract classification & confidence score, robustly.

    This approach:
      1) Cleans the response (optional).
      2) Uses a regex to find classification lines (e.g. "Classification: undersampled").
      3) Standardizes the classification to a known set by mapping synonyms to canonical labels.
      4) Extracts the confidence score via:
          - spaCy-based search for a numeric token after "confidence"
          - fallback regex search for "Confidence ...: [0-9.]+"
      5) Gracefully handles missing data or parse failures.

    Returns a dictionary with:
      {
        "classification": <str or "unknown">,
        "confidence_score": <float>
      }
    """
    def preprocess_response(raw_text):
        """
        Cleans and normalizes the response text.
        Add or remove steps here as needed.
        """
        if not raw_text:
            return ""

        cleaned = raw_text.replace("**", " ").replace("\n", " ").strip()

        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned


    clean_response = preprocess_response(response)


    parsed_result = {
        "classification": "unknown",
        "confidence_score": 0.0
    }


    classification_synonyms = {
        "motion": "motion corrupted",
        "image corruption": "motion corrupted",
        "undersample": "undersampled",
        "undersampled": "undersampled",
        "under sampled": "undersampled",
        "noise": "noisy",
        "noisy": "noisy",
        "no corruption": "no corruption",
        "none": "no corruption"
    }


    classification_regex = re.compile(
        r'classification.*?:\s*([\w\s\-]+)',                                                                 
        re.IGNORECASE
    )
    class_match = classification_regex.search(clean_response)
    if class_match:
        raw_class = class_match.group(1).lower().strip()                                             

        final_classification = None

        for key, canonical_value in classification_synonyms.items():
            if key in raw_class:
                final_classification = canonical_value
                break


        if final_classification:
            parsed_result["classification"] = final_classification
        else:


            if "motion corrupt" in raw_class:
                parsed_result["classification"] = "motion corrupted"
            else:
                parsed_result["classification"] = raw_class


    if nlp is not None:
        try:
            doc = nlp(clean_response)
            found_conf = False

            for i, token in enumerate(doc):
                if token.text.lower() in ["confidence", "confidence_score", "confidencelevel", "confidencelevel:", "confidence:", "confidence_score:", "confidencelevel"]:

                    for next_token in doc[i+1:]:

                        if next_token.like_num:
                            parsed_result["confidence_score"] = float(next_token.text)
                            found_conf = True
                            break

                        elif next_token.is_punct:
                            break
                if found_conf:
                    break
        except Exception as e:
            print(f"[Warning] spaCy parsing failed: {e}")


    if parsed_result["confidence_score"] == 0.0:
        conf_regex = re.compile(
            r'confidence\s*(?:score|level)?\s*[:\-]?\s*\[?([0-9]+(?:\.[0-9]+)?)\]?',                           
            re.IGNORECASE
        )
        conf_match = conf_regex.search(clean_response)
        if conf_match:
            try:
                parsed_result["confidence_score"] = float(conf_match.group(1))
            except ValueError:
                pass                                      


    return parsed_result


def clean_response(response: str) -> str:

    return re.sub(r"[\*\_]+", "", response)

def parse_evaluator_response(response: str, evaluator_type: str = "radiologist") -> dict:
    """
    Parses the evaluator's response (Radiologist, Assistant, or Principal Investigator)
    into a structured dictionary using multiple regex patterns.

    For Radiologist responses, it extracts:
      - classification (e.g., using "Type of corruption", "Evaluated classification")
      - confidence_score (e.g., "Confidence Score")
      - reasoning/justification (e.g., "Justification", "Reasoning behind")
      - recommended_model (e.g., "Recommended Model", "Correction Model")

    For Principal responses, it extracts:
      - final_classification (e.g., "Final Classification")
      - agreement (e.g., "Agreement with Assistants", "Agreement with Radiologist")
      - confidence_score (e.g., "Final Confidence Score")
      - recommended_model (e.g., "Final Recommended Model")
      - reasoning (e.g., "Final Justification")

    Returns:
      dict: A dictionary containing the parsed evaluation fields with default fallbacks.
    """
    result = {}
    try:
        cleaned_response = clean_response(response)
    except Exception as e:
        print("Error cleaning response:", e)
        return result

    lines = cleaned_response.strip().splitlines()


    patterns = {
        "classification": [
            r"(?:type of corruption|evaluated classification|classification)\s*:\s*(.+)",
            r"(?:final classification)\s*:\s*(.+)"
        ],
        "confidence_score": [
            r"confidence score\s*:\s*([0-9\.]+)",
            r"final confidence score\s*:\s*([0-9\.]+)"
        ],
        "reasoning": [
            r"(?:justification|reasoning(?: behind (?:the )?classification)?)\s*:\s*(.+)",
            r"(?:final justification)\s*:\s*(.+)",
            r"2\.\s*reasoning\s*:\s*(.+)",
            r"(?:explanation|analysis|observations|reasoning behind the assessment)\s*:\s*(.+)",
            r"(?:reasoning|justification)\s*:\s*\n+(.+)"                                      
        ],
        "recommended_model": [
            r"(?:recommended model|correction model|model recommendation)\s*:\s*(.+)",
            r"(?:final recommended model)\s*:\s*(.+)"
        ],
        "agreement_with_assistants": [
            r"agreement with assistant(?: responses)?\s*:\s*(.+)",
            r"alignment with assistants\s*:\s*(.+)",
            r"(?:did the assistants[’']? classifications align with expert MRI knowledge\?)\s*(yes|no|partially)"
        ],
        "agreement_with_radiologist": [
            r"agreement with radiologist\s*:\s*(.+)",
            r"alignment with radiologist\s*:\s*(.+)",
            r"(?:did the radiologist provide a strong justification\?)\s*(yes|no|partially)"
        ]
    }


    if evaluator_type == "radiologist" or evaluator_type == "assistant":
        field_patterns = {
            "classification": patterns["classification"],
            "confidence_score": patterns["confidence_score"],
            "reasoning": patterns["reasoning"],
            "recommended_model": patterns["recommended_model"]
        }
    else:                          
        field_patterns = {
            "final_classification": patterns["classification"],
            "confidence_score": patterns["confidence_score"],
            "reasoning": patterns["reasoning"],
            "recommended_model": patterns["recommended_model"],
            "agreement_with_assistants": patterns["agreement_with_assistants"],
            "agreement_with_radiologist": patterns["agreement_with_radiologist"]
        }


    for line in lines:
        cleaned_line = re.sub(r"^[\-\*\•\→\s]+", "", line).strip()
        if ':' not in cleaned_line:
            continue

        for field, regex_list in field_patterns.items():
            if field in result:
                continue                                    
            for regex in regex_list:
                match = re.search(regex, cleaned_line, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if field == "confidence_score":
                        try:
                            result[field] = float(value)
                        except ValueError:
                            result[field] = 0.5
                    else:
                        result[field] = value
                    break


    if evaluator_type == "radiologist" or evaluator_type == "assistant":
        result.setdefault("classification", "unknown")
        result.setdefault("confidence_score", 0.5)
        result.setdefault("reasoning", "No detailed justification provided.")
        result.setdefault("recommended_model", "unknown")
    else:                          
        result.setdefault("final_classification", "unknown")
        result.setdefault("agreement_with_assistants", "unknown")
        result.setdefault("agreement_with_radiologist", "unknown")
        result.setdefault("confidence_score", 0.5)
        result.setdefault("recommended_model", "unknown")
        result.setdefault("reasoning", "No detailed justification provided.")

    return result


def extract_reasoning(response_text):
    """
    Extracts the justification (reasoning) section from an AI response robustly.
    Handles cases where the reasoning is structured with numbered lists or various markers.

    Parameters:
        response_text (str): The raw text response from the AI.

    Returns:
        str: The extracted reasoning, or a default message if not found.
    """

    patterns = [

        r"\*\*2\.\s*Reasoning behind the classification\*\*:\s*(.*?)(\n\n|$)",

        r"2\.\s*Reasoning behind the classification\s*:\s*(.*?)(\n\n|$)",

        r"\*\*2\.\s*Reasoning\*\*:\s*(.*?)(\n\n|$)",

        r"2\.\s*Reasoning\s*:\s*(.*?)(\n\n|$)",

        r"(?i)reasoning\s*:\s*(.*?)(\n|$)"
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()

            reasoning_text = re.sub(r"^[\-\*\•\s]+", "", reasoning_text)
            if reasoning_text:
                return reasoning_text
    return "No reasoning provided."

