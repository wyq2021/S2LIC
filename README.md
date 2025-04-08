# S2LIC: Learned Image Compression with the SwinV2 Block, Adaptive Channel-wise and Global-inter Attention Context

[**📄 Paper on arXiv**](https://arxiv.org/abs/2403.14471)

---

## 🧠 Architectures

### Overall Framework
![Overall Framework](./assets/arch.png)

### Proposed Entropy Model
![Entropy Model](./assets/entropy.png)
![ACGC](./assets/ACGC.png)

---

## 📊 Evaluation Results

### RD Curves on Kodak
![RD Curve](./assets/kodak.png)

### Visual Quality Comparisons
![Visual Comparison](./assets/visual.png)

---

✅ TODO List
🔧 Code & Features
 Upload training and inference scripts

 Provide pretrained models (links and checkpoints)

 Add SwinV2-based encoder and decoder modules

 Implement ACGC (Adaptive Channel-wise and Global-inter Context) entropy model

 Modularize entropy coding and hyperprior network

 Add command-line tools for:

 Encoding/decoding images

 Evaluating metrics (PSNR, MS-SSIM, bpp)

📚 Documentation
 Add detailed setup instructions (dependencies, installation, environment)

 Provide training logs and tips

 Add explanation for model architecture

 Write usage examples (scripts + CLI)

📁 Dataset
 Include download instructions for benchmark datasets (Kodak, CLIC, etc.)

 Add preprocessing scripts

📈 Evaluation
 Add RD curve plotting script

 Add qualitative comparison scripts

## 📄 Citation

If you find our project useful, please cite:

```bibtex
@article{wang2024s2lic,
  title={S2LIC: Learned Image Compression with the SwinV2 Block, Adaptive Channel-wise and Global-inter Attention Context},
  author={Wang, Yongqiang and Liang, Feng and Liang, Jie and Fu, Haisheng},
  journal={arXiv preprint arXiv:2403.14471},
  year={2024}
}
```
---
## 🙏 Acknowledgement
This work is built on top of the following projects:

CompressAI (https://github.com/InterDigitalInc/CompressAI)

MLIC (https://github.com/JiangWeibeta/MLIC)

TIC (https://github.com/lumingzzz/TIC)

We thank the authors for sharing their excellent work.


