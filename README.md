<div align="center">

# Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation

### Benito Buchheim, Max Reimann, Jürgen Döllner
*Hasso Plattner Institute, University of Potsdam*

</div>

---

Welcome to the official code repository for our paper:

**"Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation"**

In this repository, we provide the implementation to control human shape and pose in pretrained text-to-image diffusion models using a 3D human parametric model (SMPL). This includes our domain adaptation technique that maintains visual fidelity while providing fine-grained control over human appearance.

---

## 🔗 [Project Page](https://ivpg.github.io/humanLDM)

## Table of Contents
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Running Sample Inference](#running-sample-inference)
- [Example Outputs](#example-outputs)
- [Citation](#citation)

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/benbuc/HumanLDMControl.git
   cd HumanLDMControl
   ```

2. **Set Up Conda Environment**
   Ensure you have Conda installed. Then, run:
   ```bash
   conda env create -f environment.yaml
   conda activate humanLDM
   ```

---

## Model Setup

1. **Download the Pretrained Model**  
   Download the model from [this link](https://drive.google.com/file/d/1r9W1GeO4iUVYD1fNWyEjI36ODpbfPL2e/view?usp=sharing) and extract it to the `checkpoints/` directory.:
   ```bash
   cd checkpoints/
   gdown --id 1r9W1GeO4iUVYD1fNWyEjI36ODpbfPL2e -O attribute_guidance.zip
   unzip attribute_guidance.zip
   ```

2. **Check Directory Structure**  
   Ensure the extracted files are in the following structure:
   ```
   checkpoints/
    └── attribute_guidance/
        ├── controlnet/
        │   └── ...
        └── smpl_embedder/
            └── ...
   ```

---

## Running Sample Inference

We compiled a very basic sample script on how to run inference using our pretrained model. To run the script, execute the following command:

```bash
python run_sample_inference.py
```

This will reconstruct the teaser images from our paper to the file `sample_output.png`.

---

## BibTeX

```bibtex
@inproceedings{buchheim2025controlling,
  author    = {Buchheim, Benito and Reimann, Max and D{\"o}llner, J{\"u}rgen},
  title     = {Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
}
```

---

## Acknowledgments
Our work "Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation" was partially funded by the German Federal Ministry of Education and Research (BMBF) through grants 01IS15041 – “mdViPro” and 01IS19006 – “KI-Labor ITSE”.


