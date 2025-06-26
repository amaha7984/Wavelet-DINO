# Forensic Detection of AI-Generated Aerial Imagery for Securing Remote Sensing Cyber-Physical Systems

<p align="center">
 <img src="./docs/Wavelet-DINO_Architecture.png" alt="Preview" width="95%" />
</p>


## Code
Official implementation of the chapter: "Forensic Detection of AI-Generated Aerial Imagery for Securing Remote Sensing Cyber-Physical Systems"

This repository introduces **Wavelet-DINO**, a new forensic detection framework that synergizes self-supervised semantic representations from DINOv2 with frequency-aware features derived via 2D Haar Discrete Wavelet Transform (DWT) for robust detection of generated aerial imagery.

### Overview
The implementation has two components:
1. Aerial Image Generation with Generative Models  
   
   Various generative model families: GANs, Diffusion, and Autoregressive, were applied to Aerial Imagery. Please refer to [`Generative_Methods&Dataset`](Generative_Methods&Dataset).

3. Wavelet-DINO Forensic Detection
   
   Please see below for code execution on training and testing
   
## Installation

- Clone the repository:
   ```bash
   git clone https://github.com/amaha7984/Wavelet-DINO.git
   cd Wavelet-DINO
   ```
- Create a Python virtual environment (optional but recommended)
  ```bash
   python3.9 -m venv myvenv
   source myvenv/bin/activate
  ```
- Install required dependencies and packages
  ```bash
  pip3.9 install -r requirements.txt
  ```
- Train the Wavelet-DINO:
```bash
python3.9 train.py --train_path ./datasets/train/ --val_path ./datasets/val/ --epochs 100 --batch_size 64
```
The model's weight will be stored at `./saved_models/`.

- Trained Weights:
  Trained weights can be downloaded from [Model_Weights](https://drive.google.com/drive/folders/1yGBiXMN9OpDnypIxW3xIe9ZGQwmtjhpt?usp=drive_link)

- Test the Wavelet-DINO:
```bash
python3.9 test.py --test_path /path/to/test_dataset/ --checkpoint_path /path/to/saved_models/weight.pth
```

### Acknowledgment

This work builds upon the ideas presented in the paper:  
[**DINOv2: Learning Robust Visual Features without Supervision**](https://arxiv.org/pdf/2304.07193), published by [Meta AI Research](https://github.com/facebookresearch/dinov2).  
We gratefully utilize the open-source DINOv2 backbone available through the 🤗 Hugging Face Transformers library:
```from transformers import Dinov2ForImageClassification
```
Special thanks to the authors and the community for their excellent work and open-source contributions.
