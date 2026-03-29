import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import dropbox
import h5py
import scipy.io
import nibabel as nib
import pydicom
from PIL import Image
import re
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


def parse_gpt4o_response(response):
    """
    Parses GPT-4o's response to extract classification and confidence.

    Parameters:
    - response: String response from GPT-4o.

    Returns:
    - classification: Extracted classification label.
    - confidence: Extracted confidence score as float.
    """
    confidence = 0.0

    confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except ValueError:
            confidence = 0.0

    return confidence

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

