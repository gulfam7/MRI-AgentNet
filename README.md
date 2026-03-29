# MRI-AgentNet

**MRI-AgentNet** is the implementation repository accompanying the paper **“MRI-AgentNet: A Vision Language Models-Based
Multi-Agent AI System for Solving Inverse Problems in MRI”**, presented at **ICAD 2026**.

The project explores a multi-agent AI workflow for **MRI corruption analysis and correction planning**, combining large multimodal models, expert-style evaluation, and downstream restoration model selection. Given an MRI input, the system first identifies whether the scan is in image space or k-space, classifies the corruption type, aggregates judgments from multiple AI evaluators, and then routes the case toward a specialized correction model.

## Overview

MRI-AgentNet is built around the idea that difficult medical image quality decisions benefit from **structured collaboration** rather than a single model response. In this repository, GPT-4o, Gemini, a radiologist-style evaluator, and a principal-investigator-style evaluator are orchestrated into a staged decision pipeline. The result is not just a label, but a richer artifact that includes:

- corruption classification
- reasoning
- confidence estimation
- recommended restoration model
- correction plan

The implementation also includes a lightweight **meta-learning module** that learns from agreement and disagreement patterns across evaluators to predict the most appropriate correction pathway.

## Core Idea

The repository operationalizes MRI corruption assessment as a coordinated decision process:

1. An input MRI scan is loaded from common research and clinical formats.
2. The scan is normalized and converted to a shareable image representation.
3. A multimodal model determines whether the input is in image space or k-space.
4. If needed, k-space data is transformed into image space for downstream reasoning.
5. Multiple AI agents classify the corruption type and propose a correction strategy.
6. A radiologist-style evaluator reviews the candidate judgments.
7. A principal-investigator-style evaluator makes the final arbitration decision.
8. A restoration model is selected and can be executed for correction.

This design is aimed at corruption categories such as:

- motion corruption
- undersampling / aliasing
- noise
- no corruption / corruption-free cases

## Repository Highlights

- Multi-agent MRI reasoning pipeline built on OpenAI and Gemini interfaces
- Support for multiple MRI data formats including `.mat`, `.h5`, `.npy`, `.nii`, `.nii.gz`, `.dcm`, and image files
- Image-space / k-space pre-classification
- Few-shot radiologist-style prompting for expert-like review
- Rule-based and meta-learning-based decision flows
- Model routing into restoration backends under [`models/`](/Volumes/Gulfam/D%20Drive%20Lab/MRI-AgentNet/models)
- Utility modules for parsing, preprocessing, confidence extraction, and model selection

## Architecture

```text
MRI Input
  -> Data Loader + Preprocessing
  -> Image-space / K-space Classification
  -> If needed: K-space to Image-space Conversion
  -> Parallel Corruption Assessment
       - GPT-4o assistant
       - Gemini assistant
  -> Radiologist-style Evaluation
  -> Principal Investigator-style Arbitration
  -> Model Selection
       - Motion correction
       - Denoising
       - Reconstruction
  -> Optional Restoration Inference
```

## Code Structure

```text
MRI-AgentNet/
├── agent_multi_meta_learning.py
├── agent_multi_eval_rule_based.py
├── model_selection/
│   ├── data_generation.py
│   ├── meta_learning.py
│   ├── meta_training_data.json
│   └── testing_meta.py
├── models/
│   ├── README.md
│   ├── test.py
│   ├── train.py
│   └── ...
├── utils/
│   ├── data_processing.py
│   ├── data_processing_confidence.py
│   ├── few_shot_gpt4o.py
│   ├── gemini_interface_confidence.py
│   ├── gpt4o_interface.py
│   ├── model_selector.py
│   ├── plan_parser.py
│   └── ...
├── data/
├── checkpoints/
└── requirements.txt
```

## Main Entry Points

### `agent_multi_meta_learning.py`

This is the primary end-to-end agent pipeline in the current repository. It combines:

- GPT-4o-based image-space and corruption analysis
- Gemini-based secondary reasoning
- radiologist-style few-shot evaluation
- principal-investigator-style final decision making
- downstream model selection
- restoration inference through the local model stack

### `agent_multi_eval_rule_based.py`

This script provides a more direct evaluation flow with rule-based aggregation logic. It is useful for simpler experiments, debugging, and comparisons against the meta-learning-enhanced approach.

### `model_selection/meta_learning.py`

This module defines a compact neural meta-model that consumes encoded evaluator decisions and predicts the final restoration model class. It represents the repository’s learning-based arbitration component.

## Supported Data Modalities

The preprocessing utilities support several common MRI storage formats:

- MATLAB `.mat`
- HDF5 `.h5` / `.hdf5`
- NumPy `.npy`
- NIfTI `.nii` / `.nii.gz`
- DICOM `.dcm`
- standard image files such as `.png`, `.jpg`, and `.jpeg`

If the data is complex-valued or multi-coil, the utility layer includes normalization and coil-combination helpers before image export.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies listed in [`requirements.txt`](/Volumes/Gulfam/D%20Drive%20Lab/MRI-AgentNet/requirements.txt) include:

- `openai`
- `google-generativeai`
- `torch`
- `numpy`
- `nibabel`
- `pydicom`
- `h5py`
- `dropbox`
- `matplotlib`
- `bert-score`

## Environment Variables

Before running the project, configure the external service credentials through environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export DROPBOX_ACCESS_TOKEN="your-dropbox-token"
```

The repository previously contained hardcoded credentials; those should now be provided only through environment variables before any public release.

## Running the Pipeline

To launch the multi-agent pipeline:

```bash
python agent_multi_meta_learning.py
```

To run the rule-based evaluation variant:

```bash
python agent_multi_eval_rule_based.py
```

Both scripts currently open a file picker for MRI input selection and then request a prompt from the user.

## Meta-Learning Workflow

The `model_selection/` directory contains a small research workflow for training and testing the learned arbitration layer:

Generate synthetic training data:

```bash
cd model_selection
python data_generation.py
```

Train the meta-model:

```bash
cd model_selection
python meta_learning.py
```

Evaluate the trained model on edge cases:

```bash
cd model_selection
python testing_meta.py
```

## Restoration Models

The restoration stack lives under [`models/`](/Volumes/Gulfam/D%20Drive%20Lab/MRI-AgentNet/models), which appears to be adapted from a CycleGAN / image-to-image translation codebase and is used here as the correction backend for:

- MRI motion correction
- MRI denoising
- MRI reconstruction / undersampling recovery

Please see [`models/README.md`](/Volumes/Gulfam/D%20Drive%20Lab/MRI-AgentNet/models/README.md) for model-specific details inherited from that framework.

## Research Positioning

MRI-AgentNet is best understood as a **medical imaging decision-and-routing framework** rather than only a single restoration model. Its contribution is the orchestration of:

- multimodal MRI understanding
- evaluator disagreement modeling
- expert-style review stages
- confidence-aware correction planning
- final restoration model routing

This makes the repository useful both for:

- AI-assisted MRI quality assessment research
- corruption-aware restoration workflows
- multi-agent medical AI experimentation
- human-in-the-loop or expert-inspired evaluation pipelines

## Reproducibility Notes

The current repository reflects an active research codebase. A few implementation details are worth knowing before publishing or reproducing experiments:

- some paths inside the scripts are still configured with local Windows-style defaults
- the main scripts depend on external APIs and Dropbox hosting for image exchange
- restoration behavior depends on local checkpoint availability under `checkpoints/` and `models/checkpoints/`
- the repository includes both rule-based and meta-learning-oriented variants of the decision flow

If you plan to release this publicly, it is a good idea to also add:

- a `.gitignore` for local artifacts and credentials
- a `.env.example`
- checkpoint download instructions
- dataset provenance notes
- citation metadata once the final bibliographic record is available

## Citation

If you use this repository in academic work, please cite the **MRI-AgentNet** paper from **ICAD 2026**.

```bibtex
@inproceedings{mri_agentnet_icad_2026,
  title={MRI-AgentNet},
  booktitle={ICAD},
  year={2026}
}
```

You can replace this placeholder entry with the final author list, page numbers, DOI, and publisher metadata once the official proceedings citation is available.

## Acknowledgments

This repository integrates:

- the MRI-AgentNet multi-agent decision workflow
- custom preprocessing and parser utilities for MRI corruption analysis
- a CycleGAN-based restoration backend adapted for MRI correction experiments

## Contact and Release Note

If you are preparing this repository for GitHub release, the README is now structured to serve as both:

- a project landing page for general readers
- a research companion page for the ICAD 2026 paper

For a stronger public release, the next best step would be adding badges, sample results figures, and a short “expected output” section with example MRI inputs and restoration outputs.
