<div align="center">

# 🎮 Make-A-Game (MAG): A Novel Paradigm for Interactive Game Rendering

<p align="center">
  <a href="https://github.com/YMlinfeng/MAG"><img src="https://img.shields.io/badge/⭐-STAR-blue?style=flat-square&logo=github" alt="GitHub Stars"></a>
  <a href="https://arxiv.org/abs/2412.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/YMlinfeng/MAG"><img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow?style=flat-square&logo=huggingface" alt="Hugging Face"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square&logo=mit" alt="License"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python" alt="Python"></a>
</p>

<p align="center">
  <b>🔥 The First DiT-based Interactive Game Video Generation Model with Unified Spatial & Non-Spatial Control </b>
</p>

<!-- <p align="center">
  <img src="assets/MAG_teaser.gif" width="85%" alt="MAG Teaser">
  <br>
  <em>Figure: MAG achieves real-time, interactive game video generation with unified control over environment semantics and entity actions, demonstrating superior stability compared to prior art.</em>
</p> -->

<p align="center">
  <strong><a href="#installation">Installation</a></strong> •
  <strong><a href="#quick-start">Quick Start</a></strong> •
  <strong><a href="#training-pipeline">Training Pipeline</a></strong> •
  <strong><a href="#evaluation">Evaluation</a></strong> •
  <strong><a href="#experimental-results">Results</a></strong> •
  <strong><a href="#todo-roadmap">TODO</a></strong>
</p>

</div>


## More content will be updated soon！

## Abstract

Driven by the growing demand for immersive and personalized gaming experiences, existing game video generation models often lack robust multi-modal control and suffer from spatial misalignment or autoregressive drift, thereby necessitating a more advanced real-time interactive solution. Make-A-Game (MAG) represents the first DiT-based interactive game video generation model that unifies spatial and non-spatial alignment control signals within the UC-3DMMAttn module and employs Action Prompt Blocks (APBs) for precise, real-time manipulation of in-game entities.

## demo video
- [ ] Coming soon

##  Project Overview

**Make-A-Game (MAG)** represents a transformative breakthrough in interactive neural game simulation, addressing fundamental limitations in existing video generation paradigms. While contemporary models such as [Sora](https://openai.com/sora), [CogVideoX](https://github.com/THUDM/CogVideo), and [OpenSora](https://github.com/hpcaitech/Open-Sora) have demonstrated remarkable capabilities in general-purpose video synthesis, they remain inadequate for interactive gaming scenarios due to three critical challenges:

1. **Control Signal Heterogeneity**: Spatially-aligned controls (e.g., segmentation masks, edge maps as in [ControlNet](https://github.com/lllyasviel/ControlNet)) conflict with non-spatially-aligned controls (e.g., IP-Adapter style semantic conditioning), requiring separate processing branches that complicate architecture and degrade performance.

2. **Autoregressive Drift**: Frame-by-frame generation accumulates errors, causing catastrophic background deformation and entity morphing over time—a phenomenon extensively documented in [GameNGen](https://gamengen.github.io/) and [DIAMOND](https://arxiv.org/abs/2407.15881).

3. **Real-time Entity Manipulation**: Existing models lack mechanisms for fine-grained, real-time control of in-game characters and objects while preserving environmental consistency.

MAG overcomes these challenges through a novel **DiT-based backbone + lightweight branch paradigm**, introducing:
- **Unified Control 3D MMDiT Attention (UC-3DMMAttn)**: Seamlessly integrates spatial and non-spatial control signals within a single attention mechanism, eliminating branch conflicts.
- **Action Prompt Block (APB)**: A plug-and-play module enabling real-time entity manipulation via keyboard inputs, decoupling character dynamics from scene generation.
- **Three-Stage Curriculum Training**: Progressive fine-tuning strategy transferring from general image generation (FLUX.1-dev) → video coherence (CogVideoX-style) → action control (APB), ensuring stable, high-fidelity outputs.


### Key Innovations

| Component | Innovation | Technical Contribution |
|-----------|------------|------------------------|
| **UC-3DMMAttn** | Unified spatial/non-spatial control | Bias-controlled attention mechanism (Δγ) |
| **APB** | Real-time entity manipulation | Gated feature modulation with scale-shift normalization |
| **Three-Stage Training** | Progressive skill acquisition | Foundation → Video → Action specialization |
| **WFVAE Encoding** | Efficient video compression | Multi-level Haar wavelet decomposition |


### Key Achievements

### Comparative Analysis with Game-Specific Models

| Model | Architecture | Real-time | HD (720P) | Unified Control | Autoregressive Drift |
|-------|--------------|-----------|-----------|-----------------|---------------------|
| Genie | Transformer | ❌ | ❌ | ❌ | High |
| MarioVGG | UNet | ❌ | ❌ | ❌ | Medium |
| GameNGen | UNet | ✅ | ❌ | ❌ | High |
| Oasis | Transformer | ✅ | ✅ | ❌ | Low |
| MAG (Ours) | DiT | ✅ | ✅ | ✅ | Very Low |

### Ablation Studies

#### Environmental Control Ablation

| Method | TF↑ | FVD↓ | FID↓ | SC↑ | BC↑ |
|--------|-----|------|------|-----|-----|
| w/o APB | 97% | 1518 | 378.6 | 61% | 27% |
| MAG (Full) | 97% | 917 | 215.8 | 63% | 84% |

#### Action Control Ablation

| Method | TF↑ | MS↑ | DD↑ | SR↑ | Color↑ |
|--------|-----|-----|-----|-----|--------|
| w/o control | 96% | 95% | 99% | 11% | 41% |
| w/o spatially aligned | 97% | 98% | 98% | 59% | 42% |
| MAG (Full) | 97% | 98% | 99% | 63% | 68% |


### Quantitative Results (VBench++ Protocol)
| Metric | MAG | Gen-3 | CogVideoX-5B | Improvement |
|--------|-----|-------|--------------|-------------|
| **Spatial Relations (SR↑)** | **63.50%** | 24.40% | 62.10% | **+2.3%** |
| **Subject Consistency (SC↑)** | **86.20%** | 78.54% | 75.12% | **+14.6%** |
| **Background Consistency (BC↑)** | **84.50%** | 76.11% | 70.14% | **+20.5%** |
| **Overall Consistency (OC↑)** | **90.70%** | 88.01% | 85.20% | **+6.5%** |
| **Temporal Flickering (TF↑)** | **97.20%** | 97.06% | 96.17% | **+0.1%** |
| **Motion Smoothness (MS↑)** | **98.20%** | 96.55% | 96.47% | **+1.7%** |

*Results averaged over 50 samples (60 frames each, 720p) on VBench++ protocol.*





##  Installation & Usage

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 16GB | 24GB+ |
| CUDA | 11.7 | 12.1+ |
| Python | 3.9 | 3.10+ |
| PyTorch | 2.0.0 | 2.1.0+ |

### Installation Steps

```bash
# Clone repository
git clone https://github.com/YMlinfeng/MAG
cd MAG

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust based on your CUDA version)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional scientific computing libraries
pip install pywavelets lpips torchmetrics
```

## Development Roadmap
### Current Implementation Status

| Component | Status | Completion | Notes |
|-----------|--------|------------|-------|
| Core Architecture | ✅ Complete | 100% | UC-3DMMDiT + APB implemented |
| Training Pipeline | ✅ Complete | 100% | Three-stage training operational |
| Evaluation Framework | ✅ Complete | 100% | VBench++ metrics integrated |
| Documentation | 🟡 Partial | 70% | API docs in progress |
| Pre-trained Models | 🟡 Partial | 60% | Stage 1 & 2 models available |
| Demo Interface | 🔄 In Progress | 40% | Gradio app under development |

#### Q4 2025
- [ ] Release pre-trained models for all three stages
- [ ] Web demo with real-time interaction
- [ ] Comprehensive API documentation
- [ ] Colab notebook for easy experimentation
- [ ] Integration with Diffusers library
- [ ] Support for 1080p generation
- [ ] Multi-player game simulation
- [ ] Extended action vocabulary (50+ actions)

#### Q1 2026
- [ ] Real-time streaming support
- [ ] Plugin for game engines (Unity/Unreal)
- [ ] Mobile optimization research
- [ ] Community model zoo
- [ ] Model quantization for deployment




## Citation & Acknowledgments

### Citation

We hope to be accepted by ICASSP2026!
<!-- If you use MAG in your research, please cite our paper:

@inproceedings{meng2025mag,
  title={Make-A-Game: A Novel Paradigm for Interactive Game Rendering},
  author={Meng, Zijie and Che, Jinming and Wei, Bingcai and Cao, Xixin},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
} -->

### Acknowledgments

This work was supported by Huawei Technologies Co., Ltd., under the research project titled "Brain-inspired Visual Processing Technologies". We thank the OpenSora and CogVideoX teams for their foundational work that inspired aspects of our architecture.

### License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions from the research community! Please see our Contributing Guidelines for details on:
- Code style and standards
- Pull request process
- Issue reporting
- Development setup
