*S2LIC: Learned Image Compression with the SwinV2 Block, Adaptive Channel-wise and Global-inter Attention ContextMulti-Reference Entropy Model for Learned Image Compression* [[Arxiv](https://arxiv.org/abs/2403.14471)]

## Architectures
The overall framework.

<img src="./assets/arch.png"  style="zoom: 33%;" />

The proposed entropy model.

<img src="./assets/entropy.png"  style="zoom: 33%;" />
<img src="./assets/ACGC.png"  style="zoom: 33%;" />

## Evaluation Results
RD curves on Kodak.

<img src="./assets/kodak.png"  style="zoom: 33%;" />

visual.
<img src="./assets/visual.png"  style="zoom: 33%;" />

## Citation
If you find our project useful, please cite the following paper.
```
@article{wang2024s2lic,
  title={S2LIC: Learned Image Compression with the SwinV2 Block, Adaptive Channel-wise and Global-inter Attention Context},
  author={Wang, Yongqiang and Liang, Feng and Liang, Jie and Fu, Haisheng},
  journal={arXiv preprint arXiv:2403.14471},
  year={2024}
}
```

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI). The base codec is adopted from [MLIC](https://github.com/JiangWeibeta/MLIC)/[TIC](https://github.com/lumingzzz/TIC). We thank the authors for open-sourcing their code.
