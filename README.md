# VisionSpice – Indian Spices Classification

![PyTorch](https://img.shields.io/badge/PyTorch-ML-red?style=flat-square)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Research-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Project Overview

VisionSpice is a comprehensive deep learning project for automatic classification of 19 Indian spice types using state-of-the-art computer vision techniques. The project demonstrates an iterative approach to model improvement through controlled experiments, progressing from a baseline CNN to an EfficientNet-B0 model with advanced augmentation and Squeeze-and-Excitation attention mechanisms.

**Research Focus**: Transfer learning, data augmentation, and attention mechanisms for fine-grained image classification.

## Dataset

- **Name**: Indian Spices Image Dataset (Mendeley)
- **Classes**: 19 Indian spice types
- **Total Images**: 10,991 high-quality RGB images
- **Split**: 70% training, 15% validation, 15% testing
- **Resolution**: Variable (preprocessed to 224×224)

**Download**: [Indian Spices Image Dataset (Mendeley)](https://data.mendeley.com/datasets/vg77y9rtjb/3)

See `data/README.md` for detailed dataset structure and download instructions.

## Models & Results

### Progressive Model Development

This project follows a scientific approach with four controlled experiments:

| Model | Architecture | Key Innovation | Val Acc | Test Acc | Overfitting Gap |
|-------|--------------|----------------|---------|----------|----------------|
| **VS_APP1** | Custom CNN | Baseline | 72% | 70% | 10% |
| **VS_APP2** | EfficientNet-B0 | Transfer Learning | 88% | 87% | 6% |
| **VS_APP3** | EfficientNet-B0 | Advanced Augmentation | 91% | 90% | 0% |
| **VS_APP4** | EfficientNet-B0 + SE | Attention Mechanism | **93%** | **92%** | **0%** |

### Key Findings

1. **Transfer Learning Impact**: +17 percentage points improvement (72% → 88%)
2. **Augmentation Effectiveness**: Eliminated overfitting gap (10% → 0%)
3. **Attention Mechanism**: +2% additional accuracy with perfect generalization
4. **Optimal Configuration**: VS_APP4 achieves state-of-the-art results

**For detailed experimental analysis, see `experiments.md`**

## Repository Structure

```
VisionSpice/
├── notebooks/
│   ├── VS_APP1_CNN.ipynb                    # Baseline CNN model training
│   ├── VS_APP2_EfficientNet.ipynb           # Transfer learning baseline
│   ├── VS_APP3_EfficientNet_Augmented.ipynb # Advanced augmentation
│   └── VS_APP4_EfficientNet_Attention.ipynb # With SE attention blocks
├── src/
│   ├── dataset_loader.py                    # Data loading and preprocessing
│   ├── cnn_model.py                         # Baseline CNN implementation
│   ├── efficientnet_model.py                # EfficientNet-B0 base model
│   ├── efficientnet_se_model.py             # SE-augmented EfficientNet
│   ├── attention_se.py                      # Squeeze-and-Excitation blocks
│   └── train_utils.py                       # Training and evaluation loops
├── data/
│   ├── README.md                            # Dataset documentation
│   └── indian_spices/                       # Dataset directory (not included)
├── results/
│   ├── confusion_matrix_app4.png            # Confusion matrix visualization
│   ├── training_curves.png                  # Training history plots
│   ├── model_comparison.png                 # Performance comparison
│   └── sample_predictions.png               # Prediction examples
├── config.yaml                              # Hyperparameters and configuration
├── experiments.md                           # Detailed experimental summary
├── inference.ipynb                          # Model inference and deployment
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # MIT License
└── README.md                                # This file
```

## Technical Details

### Model Architecture (VS_APP4)

- **Base Model**: EfficientNet-B0 (ImageNet pretrained)
- **Attention Module**: Squeeze-and-Excitation (SE) blocks
- **Input Size**: 224×224×3 RGB
- **Output Classes**: 19
- **Fine-tuning Strategy**: Last 3 blocks unfrozen

### Training Configuration

```yaml
epochs: 15
batch_size: 32
learning_rate: 0.0001
optimizer: AdamW
scheduler: Cosine annealing with warmup
data_augmentation:
  - Random rotation (15°)
  - Color jitter
  - Random affine transformations
  - MixUp and CutMix
```

See `config.yaml` for complete configuration.

## Getting Started

### Prerequisites

```bash
python >= 3.8
PyTorch >= 1.9
torchvision >= 0.10.0
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/sriram36505/VisionSpice.git
cd VisionSpice
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset
   - Download from [Mendeley](https://data.mendeley.com/datasets/vg77y9rtjb/3)
   - Extract to `data/indian_spices/`

### Training

To train the VS_APP4 model:

```bash
python -m jupyter notebook notebooks/VS_APP4_EfficientNet_Attention.ipynb
```

Or run directly in Python:
```python
from src.train_utils import train_model
from src.efficientnet_se_model import EfficientNetSE

model = EfficientNetSE(num_classes=19)
train_model(model, config='config.yaml')
```

### Inference

For predictions on new images:
```bash
python -m jupyter notebook inference.ipynb
```

## Results & Visualizations

- **Confusion Matrix**: See `results/confusion_matrix_app4.png` for per-class performance
- **Training Curves**: `results/training_curves.png` shows convergence behavior
- **Model Comparison**: `results/model_comparison.png` visualizes iteration improvements
- **Sample Predictions**: `results/sample_predictions.png` shows model outputs

## Publications & References

This project implements techniques from:

- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- Hu et al. (2019). Squeeze-and-Excitation Networks
- Cubuk et al. (2019). RandAugment: Practical automated data augmentation

## Future Work

- [ ] Evaluate on larger EfficientNet variants (B1, B2, B3)
- [ ] Ensemble methods combining multiple architectures
- [ ] Explore Vision Transformers (ViT) for comparison
- [ ] Test on mixed spice image classification
- [ ] Deploy as REST API using FastAPI/Flask
- [ ] Mobile deployment using TensorFlow Lite

## Reproducibility

This project prioritizes reproducibility:

✓ All hyperparameters in `config.yaml`
✓ Random seeds fixed for deterministic results
✓ Code well-documented with inline comments
✓ Detailed experimental logs in `experiments.md`
✓ Dataset download instructions provided
✓ Model architecture code included

## Author

**Sriram R**  
Computer Science Student | Machine Learning Researcher  
[GitHub](https://github.com/sriram36505)

## License

This project is licensed under the MIT License - see `LICENSE` file for details.

## Acknowledgments

- Indian Spices Dataset authors (Mendeley)
- PyTorch and Computer Vision communities
- NUS School of Computing for research guidance

---

**Last Updated**: January 2026  
**Total Commits**: 16+  
**Project Status**: Active Research

> This repository demonstrates best practices in machine learning research including controlled experiments, proper documentation, reproducibility, and professional presentation.
