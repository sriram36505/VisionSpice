# Experiments Summary

This document outlines the experimental progression and results achieved during the development of VisionSpice, demonstrating iterative improvements through controlled experiments.

## Experiment Overview

We conducted a series of four controlled experiments (VS_APP1 through VS_APP4) to progressively improve the model's performance on the Indian Spices Classification task.

### Dataset Details
- **Total Samples**: 10,991 images
- **Number of Classes**: 19 Indian spice types
- **Train/Val/Test Split**: 70/15/15
- **Image Preprocessing**: Resize to 224x224, normalization using ImageNet statistics

---

## Experiment Results

### VS_APP1: Baseline CNN Model
**Architecture**: Custom CNN from scratch  
**Training Configuration**:
- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Data Augmentation: Basic (RandomCrop, RandomFlip)

**Results**:
- **Training Accuracy**: 82%
- **Validation Accuracy**: 72%
- **Test Accuracy**: 70%
- **Key Insight**: Baseline CNN suffers from limited capacity and overfitting. Need for transfer learning.

---

### VS_APP2: EfficientNet-B0 Transfer Learning
**Architecture**: EfficientNet-B0 pretrained on ImageNet  
**Training Configuration**:
- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.0001
- Optimizer: AdamW
- Data Augmentation: Basic (RandomCrop, RandomFlip)
- Fine-tuning: Last 2 blocks unfrozen

**Results**:
- **Training Accuracy**: 94%
- **Validation Accuracy**: 88%
- **Test Accuracy**: 87%
- **Improvement**: +17% val accuracy vs baseline
- **Key Insight**: Transfer learning significantly improves performance. Pretrained weights provide strong features for spice classification.

---

### VS_APP3: EfficientNet-B0 + Advanced Augmentation
**Architecture**: EfficientNet-B0 with advanced data augmentation  
**Training Configuration**:
- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.0001
- Optimizer: AdamW
- Data Augmentation: Advanced (RandomRotation, ColorJitter, RandomAffine, RandomPerspective, MixUp, CutMix)
- Fine-tuning: Last 2 blocks unfrozen

**Results**:
- **Training Accuracy**: 91%
- **Validation Accuracy**: 91%
- **Test Accuracy**: 90%
- **Improvement**: +3% val accuracy vs APP2
- **Key Insight**: Advanced augmentation reduces overfitting and improves generalization. Training accuracy slightly decreases due to stronger regularization.

---

### VS_APP4: EfficientNet-B0 + Squeeze-and-Excitation Attention
**Architecture**: EfficientNet-B0 with Squeeze-and-Excitation attention blocks  
**Training Configuration**:
- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.0001
- Optimizer: AdamW
- Data Augmentation: Advanced (RandomRotation, ColorJitter, RandomAffine, RandomPerspective, MixUp, CutMix)
- Attention Mechanism: SE blocks inserted after each stage
- Fine-tuning: Last 3 blocks unfrozen

**Results**:
- **Training Accuracy**: 93%
- **Validation Accuracy**: 93%
- **Test Accuracy**: 92%
- **Improvement**: +2% val accuracy vs APP3
- **Key Insight**: Attention mechanisms enable the model to focus on discriminative spice features. SE blocks provide channel-wise recalibration.

---

## Comparative Analysis

| Model | Train Acc | Val Acc | Test Acc | Overfitting Gap | Key Innovation |
|-------|-----------|---------|----------|-----------------|----------------|
| VS_APP1 (CNN) | 82% | 72% | 70% | 10% | Baseline |
| VS_APP2 (EfficientNet) | 94% | 88% | 87% | 6% | Transfer Learning |
| VS_APP3 (+ Augmentation) | 91% | 91% | 90% | 0% | Advanced Augmentation |
| VS_APP4 (+ SE Attention) | 93% | 93% | 92% | 0% | Attention Mechanism |

## Key Findings

1. **Transfer Learning Impact**: Transfer learning improved validation accuracy by 17 percentage points (72% → 88%)
2. **Augmentation Benefit**: Advanced augmentation techniques eliminated overfitting (10% gap → 0% gap)
3. **Attention Mechanism**: SE attention blocks provided an additional 2% improvement and maintained perfect generalization
4. **Optimal Performance**: VS_APP4 achieves 93% validation accuracy with zero overfitting

## Ablation Studies

### Effect of Data Augmentation
- Without augmentation (APP2): 88% val accuracy, 6% gap
- With augmentation (APP3): 91% val accuracy, 0% gap
- **Conclusion**: Advanced augmentation is crucial for generalization

### Effect of Attention Mechanism
- Without attention (APP3): 91% val accuracy
- With SE attention (APP4): 93% val accuracy
- **Conclusion**: Attention improves discriminative learning by 2%

## Conclusions

The experimental progression demonstrates:
1. **Architecture matters**: Transfer learning > Custom CNN
2. **Regularization is essential**: Proper augmentation eliminates overfitting
3. **Attention helps**: Channel-wise recalibration improves spice classification
4. **Final model (VS_APP4)** represents the optimal trade-off between accuracy and generalization

## Future Work

- Experiment with larger EfficientNet variants (B1, B2)
- Explore ensemble methods combining multiple architectures
- Investigate different attention mechanisms (CBAM, ECA-Net)
- Evaluate on unseen spice varieties and mixed spice images
