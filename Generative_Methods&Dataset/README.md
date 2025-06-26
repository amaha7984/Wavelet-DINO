# Generative Methods for Constructing Aerial-Forensics Dataset


## Models Trained
We trained the following generative models using their official code repositories:

### Generative Adversarial Networks (GANs)
1. **ProGAN**  
   Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation," 2017. [Paper](https://arxiv.org/abs/1710.10196) • [Github](https://github.com/tkarras/progressive_growing_of_gans)
   
2. **StyleGANv2-ADA**  
   Karras et al., "Training Generative Adversarial Networks with Limited Data," 2020. [Paper](https://arxiv.org/abs/2006.06676) • [Github](https://github.com/NVlabs/stylegan2-ada-pytorch))

3. **CycleGAN**  
   Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," 2017. [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) • [Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))

4. **AttentionGAN**  
   Tang et al., "AttentionGAN: Unpaired Image-to-Image Translation Using Attention-Guided Generative Adversarial Networks," 2023. [Paper](https://ieeexplore.ieee.org/document/9527389) • [Github](https://github.com/Ha0Tang/AttentionGAN))

5. **KAN-CUT**  
   Mahara et al., "The Dawn of KAN in Image-to-Image (I2I) Translation: Integrating Kolmogorov-Arnold Networks with GANs for Unpaired I2I Translation," 2024. [Paper](https://arxiv.org/pdf/2408.08216) • [Github](https://github.com/amaha7984/kan-cut))

### Diffusion Model
6. **DiT (Diffusion Transformer)**  
   Peebles et al., "Scalable Diffusion Models with Transformers," 2023. [Paper](https://arxiv.org/abs/2212.09748) • [Github](https://github.com/facebookresearch/DiT)

### Autoregressive Generative Model
7. **RAR (Randomized Autoregressive Generation)**  
   Yu et al., "Randomized autoregressive visual generation," 2024. [Paper](https://doi.org/10.48550/arXiv.2411.00776) • [Github](https://github.com/bytedance/1d-tokenizer)

---

### Aerial-Forensics Dataset Access

The dataset is hosted on Google Drive and can be accessed here:
[Download Aerial-Forensics Dataset](https://drive.google.com/drive/folders/1KMJ-Jwcs5JuoAZ-5JTuILCiTw_lg3te-?usp=drive_link)

> **License:** By downloading this dataset, you agree to the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

The dataset has train, val, and test folders. Inside each folder, there are two subfolders named 0_real, which contains real aerial imagery, and 1_fake, which contains generated aerial imagery.
