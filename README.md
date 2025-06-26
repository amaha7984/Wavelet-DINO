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
