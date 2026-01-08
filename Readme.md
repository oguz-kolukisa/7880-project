# Addressing Background Context Bias in Few-Shot Segmentation through Iterative Modulation (CVPR 2024)

This readme file is an outcome of the [CENG7880 (Fall 2025)](https://metu-trai.github.io/) project for reproducing a paper which does not have an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/metu-trai/Projects2025) for a complete list of all paper reproduction projects.

---

## TL;DR

- **What's Implemented**: Complete ABCB architecture with Query Prediction, Support Modulation, and Information Cleansing modules; iterative refinement (T=3); ResNet-50/101 backbones; PASCAL-5^i and COCO-20^i datasets with automatic mask generation
- **What's Trained**: COCO-20^i with ResNet-50 for both 1-shot and 5-shot (70 epochs × 20,000 episodes, ~160 GPU hours  on fold 0)
- **Results**: 1-shot: 13.36% best mIoU | 5-shot: 21.67% best mIoU (single fold)
- **Difference from Paper**: Paper reports 45.6% for 1-shot (4-fold mean); current single-fold results are lower, likely due to undocumented implementation details
- **Code Status**: Fully modular and ready for training all configurations (PASCAL/COCO, ResNet-50/101, 1/5-shot, all folds)
---

## Table of Contents

1. [Introduction](#1-introduction)
   - [Background Context Bias Problem](#background-context-bias-problem)
   - [The Proposed Solution](#the-proposed-solution)
   - [Paper Summary](#11-paper-summary)
2. [The Method and Implementation](#2-the-method-and-implementation)
   - [The Original Method](#21-the-original-method)
   - [Implementation Details and Assumptions](#22-implementation-details-and-assumptions)
3. [Experiments and Results](#3-experiments-and-results)
   - [Experimental Setup](#31-experimental-setup)
   - [How to Run the Code](#32-how-to-run-the-code)
   - [Data Layout](#data-layout)
   - [Quick Start Recipes](#quick-start-recipes)
   - [Results](#33-results)
4. [Conclusion](#4-conclusion)
5. [References](#5-references)
6. [Contact](#contact)

---

# 1. Introduction

Few-shot semantic segmentation (FSS) equips neural networks to segment objects from unseen categories using only a small number of annotated examples—commonly 1 or 5 support images paired with masks. This is especially valuable in settings where large-scale labeling is infeasible, including medical imaging, autonomous driving, and robotics. The core difficulty is enabling fast adaptation to new categories while still producing accurate, fine-grained segmentation boundaries under very limited supervision.

## The Background Context Bias Problem

Although FSS has progressed substantially, a major bottleneck remains: **background context bias**. This occurs when the support and query images contain notably different backgrounds, which can cause substantial feature mismatch and degrade segmentation.

Most FSS approaches compute prototypes from the foreground regions of the support images (as specified by masks) and then use these prototypes to steer segmentation in the query image. However, because CNN features are influenced by surrounding pixels through receptive fields, foreground representations are often “mixed” with background context. When the query background differs from the support background, the resulting foreground features can shift—even when the object category is the same.

**Illustrative example:** Suppose a support image contains a dog on grass, while the query image shows a dog on sand at the beach. Even though both contain the same type of foreground object, the extracted “dog” features in the support image may be affected by grass textures and lighting. As a result, those features may not align well with the dog in the beach scene, producing weak activation and segmentation errors.

**Concrete scenario:** A cat appears indoors on carpet in the support set but outdoors on concrete in the query. The indoor cat’s foreground features can be shaped by the surrounding lighting and textures. When transferred to the query image, these features may fail to highlight the outdoor cat because the contextual cues differ, leading to partial masks or even complete failure.

This problem is pervasive in few-shot learning but has been largely overlooked by prior work, which assumes feature extractors can produce context-invariant foreground representations—an assumption that breaks down in practice.

## The Proposed Solution: Iterative Modulation
The ABCB approach mitigates background context bias through an **iterative modulation framework** composed of three complementary modules:

1. **Query Prediction (QP)**: Produces an initial segmentation mask from the current support-guided representation and refines the mask over successive iterations.

2. **Support Modulation (SM)**: Adjusts support features to better match the query’s context by **examining query evolution patterns**, how query representations change from shallow to deep network layers. This evolution helps distinguish stable foreground cues from context-sensitive ones, enabling more targeted support alignment (e.g., via cross-attention).

3. **Information Cleansing (IC)**: Prevents error accumulation during modulation by filtering unreliable signals using **confidence-aware attention**. Low-confidence regions (often identified through entropy-based uncertainty) are suppressed so they do not contaminate the next iteration.

The model runs for **T = 3 iterations**, and each cycle updates both the predicted mask and the guidance features. This repeated refinement progressively reduces context-driven feature mismatch, leading to more accurate segmentation.

**Key Idea:** RInstead of trying to learn perfectly background-invariant features (which is difficult and often unrealistic), the method explicitly accounts for how background context distorts representations and then corrects this distortion through iterative refinement. By combining query evolution analysis with confidence-guided denoising, the correction remains both targeted and stable across iterations.

## 1.1. Paper Summary

**Publication:** *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*  
**Authors:** Zhu, H., et al.

### Key Contributions

1. **Formalization of Background Context Bias**  
   The paper provides the first systematic analysis of how background context differences between support and query images cause feature misalignment in FSS, including quantitative ablation studies demonstrating the severity of this issue.

2. **Iterative Modulation Architecture**  
   Introduces a three-stage iterative framework (Query Prediction → Support Modulation → Information Cleansing) that explicitly addresses context bias through feature alignment rather than relying solely on discriminative learning.

3. **Query Evolution Mechanism**  
   Proposes a novel method to capture how query features evolve through the network, decomposing evolution into:
   - **Pixel-wise evolution** ($E_p$): Changes in individual feature vectors from input to deep layers
   - **Structure-wise evolution** ($E_s$): Changes in pairwise feature relationships captured via histogram-based affinity analysis
   
   These evolution features serve as context-aware guidance for support modulation.

4. **Confidence-Biased Information Cleansing**  
   Develops an entropy-based confidence mechanism that identifies uncertain predictions and uses these confidence maps to weight attention during feature denoising, preventing error accumulation across iterations.

5. **State-of-the-Art Performance**  
   Achieves State-of-the-Art on few-shot segmentation datasets:
   - **PASCAL-5^i**: Consistent improvements across all 4 folds in both 1-shot and 5-shot settings
   - **COCO-20^i**: Strong performance on the more challenging 80-class dataset with 4-fold cross-validation
   
   Demonstrates particular strength in scenarios with high background diversity.

### Technical Novelty

The paper's innovation centers on **treating background context bias as a correctable misalignment** rather than an unavoidable noise source. Key technical distinctions from prior work:

- **Explicit modeling of context influence:** Uses query evolution analysis to quantify how much background context affects features, rather than hoping learned features will be inherently robust.

- **Iterative refinement with feedback:** Each iteration uses the current prediction to identify which support features need adjustment, enabling targeted corrections rather than global feature transformations.

- **Dual-stream attention design:** Separates the process of extracting context information (using background query features and evolution patterns) from the process of applying corrections (via cross-attention on support features).

- **Confidence-aware denoising:** Confidence-aware denoising incorporates the model’s uncertainty directly into feature refinement: high-confidence, reliable regions guide the denoising updates, while low-confidence regions are down-weighted so they do not introduce additional noise.

As a result, the model stays robust to large background variations without requiring any background-specific supervision, and it generalizes better across a broad range of few-shot segmentation scenarios.

---

# 2. The Method and Implementation

## 2.1. The Original Method

The paper presents an iterative few-shot segmentation framework that refines the guidance extracted from support images over multiple iterations. Each iteration is composed of three components:

1. **Query Prediction (QP)**  
2. **Support Modulation (SM)**  
3. **Information Cleansing (IC)**

All components use the same backbone to create the feature maps $f_s$ and $f_q$ based on the support image $I_s$ and the query image $I_q$. The detailed network architecture is shown and explained below:

![Figure 2](images/fig2_overall_framework.png)

### 2.1.1. Query Prediction

The Query Prediction module generates a segmentation mask for the query image using a foreground guidance feature from the support image.  

Let the support and query feature maps be $f_s$ and $f_q \in \mathbb{R}^{C \times H' \times W'}$.  
The downsampled foreground mask for the support image is $G_s$.  

Using $G_s$, the method selects the foreground positions in $f_s$, treats the corresponding feature vectors as tokens, and collects them into a foreground feature set $S_{\text{QP}}^{1}$ for the first iteration.

At iteration $t$, the guidance feature set $S_{\text{QP}}^{t}$ is used in conjunction with the query feature map $f_q$ to predict the query mask. Specifically, the prediction logits $P^{t}$ are obtained as:

$$P^{t} = \phi_p\left(\text{CAT}\left(\text{AVG}\left(S_{\text{QP}}^{t}\right), f_q\right)\right)$$

where $\text{AVG}(\cdot)$ denotes the average over all tokens in $S_{\text{QP}}^{t}$, and $\text{CAT}(\cdot)$ denotes concatenation along the channel dimension. The mapping $\phi_p$ is instantiated as two successive $1 \times 1$ convolutional layers that transform the concatenated features into the prediction logits $P^{t}$.

For subsequent iterations, the guidance feature set is updated using the output of the Information Cleansing module, according to:

$$S_{\text{QP}}^{t} \leftarrow S_{\text{IC}}^{t-1}, \quad t = 2 \to T$$

Conceptually, the module acts as a prototype-based predictor: at each iteration, it uses a prototype computed from the support set to generate the query segmentation mask, then refines this prototype through the Information Cleansing step before reusing it in the next iteration.

![Figure A](images/figA_query_prediction.png)

### 2.1.2. Support Modulation

The Support Modulation step is designed to reduce the mismatch between support and query foreground features that arises from differences in background. The main idea is to extract a **query evolution feature** that describes how the query foreground changes from an input-level representation to the deep backbone representation, and then to use this information to adjust the support foreground feature so that it becomes more consistent with the query.

The evolution feature is split into two parts. The pixel-wise evolution feature $E_p$ describes how each individual foreground pixel changes between a context-independent representation at the input and a context-influenced representation at the backbone output. The structure-wise evolution feature $E_s$ summarizes how pairwise relationships or affinities among foreground pixels change between input and output, using histogram-based statistics to capture these structural shifts.

These two terms are then combined as:

$$E = E_p + E_s$$

where $E_s$ is broadcast to all foreground tokens and added to $E_p$, so that each token carries both pixel-level and structure-level evolution information.

On the query side, the method forms a set of background tokens $B$ from features at locations that are currently predicted as background, based on the prediction $P^t$. A cross-attention operation then uses $B$ as queries and the evolution feature $E$ as keys and values to compute a context representation $C$:

$$C = \text{ATT}(Q_B, K_E, V_E)$$

where ATT denotes a standard query–key–value attention block.

Finally, the support feature is modulated using this context. The model attends from the current guidance $S_{\text{QP}}^{t}$ to the context $C$ and adds the result back to the guidance:

$$S_{\text{SM}}^{t} = S_{\text{QP}}^{t} + \text{ATT}\big(Q_{S_{\text{QP}}^{t}}, K_C, V_C\big)$$

The resulting feature $S_{\text{SM}}^{t}$ is a query-aware support representation that is intended to be better aligned with the query foreground than the original $S_{\text{QP}}^{t}$.

![Figure B](images/figB_support_modulation.png)

### 2.1.3. Information Cleansing

Because Support Modulation depends on the current prediction $P^t$, errors in this prediction can introduce noise into the modulated guidance $S_{\text{SM}}^{t}$. The Information Cleansing step aims to reduce such noise using a confidence-biased attention mechanism.

The process starts by computing a second prediction $\hat{P}^{t}$. This is done by running the same Query Prediction step again, but now using the modulated feature $S_{\text{SM}}^{t}$ instead of the original guidance $S_{\text{QP}}^{t}$. From both $P^t$ and $\hat{P}^t$, the method derives entropy-based confidence maps, denoted by $C$ and $\hat{C}$. It then measures how the confidence changes by forming:

$$V = \hat{C} - C$$

which highlights pixels where the confidence has decreased.

The map $V$ is injected into an attention module that attends from a feature built from the pair $[S_{\text{QP}}^{t}, S_{\text{SM}}^{t}]$ to the query features $f_q$:

$$N = \text{softmax}\big(Q_{\phi[S_{\text{QP}}^{t}, S_{\text{SM}}^{t}]} K_{f_q} + V\big)\, V_{f_q}$$

Here, the term $V$ biases the attention weights so that regions with lower confidence receive more attention. A linear layer $\phi$ then maps $N$ into the same feature space as $S_{\text{SM}}^{t}$, and the cleaned guidance feature is obtained by subtracting this correction:

$$S_{\text{IC}}^{t} = S_{\text{SM}}^{t} - \phi(N)$$

The feature $S_{\text{IC}}^{t}$ is passed forward as the new guidance $S_{\text{QP}}^{t+1}$ in the next iteration, which closes the loop of the three-stage recurrent design.

![Figure C](images/figC_information_cleansing.png)

---

## 2.2. Implementation Details and Assumptions

This implementation makes several design choices and assumptions not explicitly specified in the paper:

### Architecture Assumptions

1. **Trainable Backbone**: The backbone (ResNet-50/101) is kept trainable rather than frozen. The paper does not explicitly state whether the backbone should be frozen, but keeping it trainable allows the model to adapt features to the few-shot segmentation task.

2. **Batch Normalization Mode**: BatchNorm layers in the backbone remain in their default PyTorch mode (switching between train/eval based on model.train()/model.eval()). The paper does not specify special handling for BatchNorm.

3. **Multi-Head Attention Configuration**: Support Modulation uses 4-head attention. The paper mentions multi-head attention but does not specify the number of heads.

4. **Token Limits**: To prevent memory overflow with large images or many foreground pixels, the implementation caps:
   - Maximum support tokens: 1024
   - Maximum foreground tokens: 512  
   - Maximum background tokens: 512
   
   When token counts exceed these limits, random sampling is used. The paper does not discuss token limiting strategies.

5. **Feature Projection Dimension**: All backbone features are projected to 256 dimensions. While the paper shows 256-d in the architecture diagram, the exact projection mechanism is not fully detailed.

### Data Processing Assumptions

1. **ImageNet Normalization**: Input images are normalized using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. The paper mentions using ImageNet-pretrained backbones but does not explicitly state the normalization scheme.

2. **Random Crop Behavior**: During training, random crops of size 473×473 (PASCAL) or 641×641 (COCO) are applied after random scaling. The paper specifies crop sizes but not the exact cropping strategy when images are smaller than the crop size.

3. **Validation Augmentation**: Validation uses deterministic resizing to crop size without random augmentation, keeping normalization. The paper does not detail validation preprocessing.

4. **Binary Mask Construction**: Support and query masks are constructed as (mask == class_id) for multi-class datasets. For datasets with instance annotations, all instances of the target class are merged into a single binary mask.

### Training Assumptions

1. **Learning Rate Scaling**: Base learning rate is not scaled with batch size in the default configuration (2e-3 for PASCAL, 5e-3 for COCO with batch sizes 16 and 8 respectively). The paper provides base LR values but does not discuss scaling strategies.

2. **Polynomial LR Schedule Parameters**: The polynomial learning rate decay uses power=0.9 as per the paper, with final learning rate going to 0 by the end of training.

3. **Optimizer Initialization**: SGD optimizer is initialized with momentum=0.9 and weight_decay=1e-4 as stated in the paper. No learning rate warmup is used (not mentioned in paper).

4. **Episode Sampling**: Training episodes are sampled randomly with the same random seed (0) for reproducibility. The paper specifies episode counts but not the sampling strategy.

### Evaluation Assumptions

1. **IoU Computation**: Binary IoU is computed from logits using threshold 0 (positive logits = foreground). The paper reports mIoU but does not specify the binarization threshold.

2. **COCO Mask Generation**: For COCO-20^i, binary segmentation masks are automatically generated from instance annotations and cached to disk. The paper uses COCO-20^i but does not describe mask preparation in detail.

3. **Fold Assignment**: Class-to-fold assignment follows the standard PASCAL-5^i and COCO-20^i splits established in prior work (e.g., PANet, HSNet). The paper references these benchmarks without reproducing the split details.


These assumptions were made based on common practices in few-shot segmentation literature, standard PyTorch conventions, and implementation details from similar methods. Any deviation from the paper's intended design may affect the reproducibility of results.

---

# 3. Experiments and Results

## 3.1. Experimental Setup

### Model Configuration
- **Architecture**: ABCB with ResNet-50 backbone (trainable)
- **Feature Dimension**: 256-d projection
- **Iterations**: T=3
- **Attention**: 4-head multi-head attention in Support Modulation
- **Token Limits**: max_support_tokens=1024, max_fg_tokens=512, max_bg_tokens=512

### Datasets
This replication focuses on **COCO-20^i** with **ResNet-50** backbone in **1-shot** setting:
- **COCO-20^i**: 80 MS-COCO categories split into 4 folds of 20 classes each
- **Folds**: Training on folds 0-3
- **Shot**: 1-shot (one support example per class)
- **Binary Masks**: Auto-generated from instance annotations and cached

### Data Augmentation
- **Training**: 
  - Random scale: 0.5–2.0
  - Horizontal flip: probability 0.5
  - Random crop: 641×641
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Validation**: 
  - Deterministic resize to 641×641
  - ImageNet normalization only

### Optimization
- **Optimizer**: SGD
- **Momentum**: 0.9
- **Weight Decay**: 1e-4
- **Learning Rate**: Base LR = 0.005
- **LR Schedule**: Polynomial decay (power=0.9)
- **Batch Size**: 8
- **Episodes**: 20,000 training episodes, 2,000 validation episodes
- **Epochs**: 70 for COCO-20^i
- **Data Loading**: num_workers=4, pin_memory=True

### Computational Requirements
- **Training Time per Fold**: ~80 hours on GPU
- **Total Training Time**: ~320 hours for all 4 folds (only 2 folds completed)
- **Hardware**: NVIDIA GPU with CUDA support

## 3.2. How to Run the Code

### Prerequisites

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

Required packages include: torch, torchvision, numpy, Pillow, scipy, tqdm, huggingface_hub (optional)

### Data Layout

Expected folder structure after dataset preparation:

```
data/
├── coco/
│   ├── images/
│   │   ├── train2017/          # COCO train images
│   │   │   ├── 000000000009.jpg
│   │   │   └── ...
│   │   └── val2017/            # COCO val images
│   │       ├── 000000000139.jpg
│   │       └── ...
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── masks/                  # Auto-generated binary masks (cached)
│   │   ├── train2017/
│   │   └── val2017/
│   └── coco20i_fold0_train.pt  # Cached class maps per fold/split
│
├── VOCdevkit/
│   └── VOC2012/
│       ├── JPEGImages/         # PASCAL images
│       ├── SegmentationClass/  # Segmentation masks
│       └── ImageSets/
│           └── Segmentation/
│               ├── train.txt
│               └── val.txt
│
└── sbd/                        # SBD (Semantic Boundaries Dataset)
    ├── img/
    ├── cls/
    └── train.txt
```

**Notes:**
- COCO masks are auto-generated from instance annotations on first run and cached
- PASCAL-5^i uses VOC2012 + SBD for extended training data
- Cached `.pt` files (class maps) speed up subsequent runs

### Quick Start Recipes

**Recipe 1: Prepare datasets only (no training)**
```bash
python src/replicate_abcb.py \
  --download \
  --prepare-only \
  --data-root ./data
```

**Recipe 2: Train COCO-20^i ResNet-50 1-shot (fold 0 only)**
```bash
python src/replicate_abcb.py \
  --datasets coco20i \
  --backbones resnet50 \
  --shots 1 \
  --folds 0 \
  --data-root ./data \
  --output-dir ./output \
  --device cuda
```

**Recipe 3: Train COCO-20^i ResNet-50 5-shot (all 4 folds)**
```bash
python src/replicate_abcb.py \
  --datasets coco20i \
  --backbones resnet50 \
  --shots 5 \
  --folds 0 1 2 3 \
  --data-root ./data \
  --output-dir ./output \
  --device cuda
```

**Recipe 4: Evaluate existing checkpoints only (no training)**
```bash
python src/replicate_abcb.py \
  --eval-only \
  --data-root ./data \
  --output-dir ./output \
  --device cuda
```

**Recipe 5: Quick debug run (1 epoch, 4 episodes)**
```bash
python src/replicate_abcb.py \
  --debug \
  --datasets coco20i \
  --backbones resnet50 \
  --shots 1 \
  --folds 0 \
  --data-root ./data \
  --output-dir ./output
```

2. **Prepare Datasets**:

Option A - Download using the script:
```bash
python src/replicate_abcb.py --download --prepare-only --data-root ./data
```

Option B - Manual download:
- **COCO**: Download COCO 2017 train/val images and annotations from [https://cocodataset.org/](https://cocodataset.org/)
  - Place in `./data/coco/images/train2017/`, `./data/coco/images/val2017/`
  - Place annotations in `./data/coco/annotations/`

### Training from Scratch

To train on COCO-20^i with ResNet-50 (1-shot) for all folds:
```bash
python src/replicate_abcb.py \
  --data-root ./data \
  --output-dir ./output \
  --device cuda \
  --log-level INFO
```

### Filtering Specific Configurations

To train on specific datasets, backbones, shots, or folds:
```bash
python src/replicate_abcb.py \
  --datasets coco20i \
  --backbones resnet50 \
  --shots 1 \
  --folds 0 1 \
  --data-root ./data \
  --output-dir ./output
```

### Evaluation Only

To evaluate existing checkpoints without retraining:
```bash
python src/replicate_abcb.py \
  --eval-only \
  --data-root ./data \
  --output-dir ./output
```

### Debug Mode

For quick testing with reduced data and iterations:
```bash
python src/replicate_abcb.py \
  --debug \
  --data-root ./data \
  --output-dir ./output
```

Debug mode overrides:
- train_episodes: 4
- val_episodes: 2
- epochs: 1
- batch_size: 2
- num_workers: 0
- max_steps: 2

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-root` | Dataset root directory | `./data` |
| `--output-dir` | Directory for checkpoints and results | `./output` |
| `--device` | Device for training (cuda/cpu) | `cuda` |
| `--download` | Download datasets using torchvision | - |
| `--debug` | Enable debug mode with limited steps | - |
| `--prepare-only` | Only prepare datasets and exit | - |
| `--eval-only` | Only evaluate existing checkpoints | - |
| `--log-level` | Logging level | `INFO` |
| `--datasets` | Filter specific datasets | `pascal5i coco20i` |
| `--backbones` | Filter specific backbones | `resnet50 resnet101` |
| `--shots` | Filter specific shots | `1 5` |
| `--folds` | Filter specific folds | `0 1 2 3` |

### Output Structure

After training, results are organized as:
```
output/
├── replication_results.json          # Aggregated mIoU results
└── coco20i_resnet50_1shot/
    ├── fold0/
    │   ├── model.pt                  # Model checkpoint
    │   └── training_metrics.json     # Training loss and validation IoU per epoch
    ├── fold1/
    │   ├── model.pt
    │   └── training_metrics.json
    └── ...
```

### Resume Training

The script automatically:
- Loads existing `replication_results.json` if present
- Skips folds that already have evaluation results
- Loads existing checkpoints instead of retraining

To force retraining, delete the relevant checkpoint or result entry.

## 3.3. Results

### Paper vs Replication Comparison

| Setting | Paper (4-fold mean) | Replication (fold 0 only) |
|---------|---------------------|---------------------------|
| **1-shot** | 45.6% mIoU | 13.36% mIoU (best epoch 36) |
| **5-shot** | 58.3% mIoU | 21.67% mIoU (best epoch 21) |

**Important Note**: Paper results represent the mean across all 4 folds, while replication results are for fold 0 only due to computational constraints. The significant performance gap suggests potential implementation differences or insufficient training convergence for this particular fold.

### Paper Results (COCO-20^i, ResNet-50, 1-shot)

From the original paper, the reported mean mIoU across all folds for COCO-20^i with ResNet-50 backbone in 1-shot setting is approximately **45.6%**.

### Replication Results (COCO-20^i, ResNet-50)

Full training (70 epochs × 20,000 episodes) has been completed for both 1-shot and 5-shot settings. Training was conducted on a single fold to validate the implementation.

**Note on Implementation Scope**: The codebase includes full implementations for both **PASCAL-5^i** and **COCO-20^i** datasets, as well as both **ResNet-50** and **ResNet-101** backbones. However, due to computational resource limitations (~80 hours per fold × 4 folds × 2 backbones × 2 datasets = 2,560 GPU hours for complete replication), training and evaluation were limited to COCO-20^i with ResNet-50 backbone only. All other configurations are implemented and ready for training when computational resources become available.

#### Training Metrics Summary

| Setting | Final Train Loss | Final Val mIoU | Best Val mIoU | Best Epoch |
|---------|-----------------|----------------|---------------|------------|
| **1-shot** | 0.2482 | 0.0884 | 0.1336 | 36 |
| **5-shot** | 0.1643 | 0.1661 | 0.2167 | 21 |

**Key Observations:**
- Training loss decreases steadily across 70 epochs for both settings
- 5-shot achieves significantly better performance (16.61% vs 8.84% final mIoU)
- Best validation performance occurs mid-training (epoch 21-36), suggesting potential overfitting in later epochs
- 5-shot shows 2.45× better final mIoU compared to 1-shot

**Computational Cost:**
- Each fold requires approximately **80 hours of GPU training**
- Total time for both 1-shot and 5-shot: **160 hours**

### Visualization

#### Qualitative Results: 1-shot Model

![1-shot Examples](images/examples_1shot.png)

*Figure: Segmentation results for the 1-shot model on four randomly selected validation examples. Each row shows: (1) Support image with ground truth mask overlay, (2) Query image to segment (unfiltered, properly denormalized), (3) Ground truth mask (green overlay), (4) Predicted mask (blue overlay), (5) Comparison overlay where green = ground truth only, red = prediction only, yellow = correct overlap.*

**Analysis of 1-shot Results:**

The 1-shot model demonstrates several characteristic behaviors:

1. **Coarse Localization**: The model successfully identifies the general location of target objects in most cases, indicating that the feature extraction and prototype matching mechanisms are functioning. However, predictions tend to be coarse and lack precise boundary delineation.

2. **Under-segmentation**: A common pattern is under-segmentation where the predicted masks capture only portions of the target object rather than the complete instance. This suggests the model struggles to propagate segmentation from high-confidence regions to entire object extents.

3. **Background Confusion**: In some examples, the model incorrectly segments background regions that share similar visual characteristics with the support object, highlighting the background context bias problem that the paper aims to address. The iterative refinement mechanism shows limited effectiveness in these cases.

4. **Low IoU Scores**: The IoU scores remain modest, reflecting the challenges of few-shot segmentation with minimal supervision. Single support examples provide insufficient guidance for capturing object appearance variations.

#### Qualitative Results: 5-shot Model

![5-shot Examples](images/examples_5shot.png)

*Figure: Segmentation results for the 5-shot model on the same randomly selected validation examples. Same visualization format as 1-shot model above. Query images are unfiltered and properly denormalized for clear visualization.*

**Analysis of 5-shot Results:**

The 5-shot model shows notable improvements over 1-shot:

1. **Improved Coverage**: With access to multiple support examples, the model achieves better object coverage and more complete segmentation masks. The additional examples help the model learn a more comprehensive representation of the target class.

2. **Better Boundary Localization**: Predictions exhibit improved boundary localization compared to 1-shot, though still fall short of precise pixel-level accuracy. The model better distinguishes object edges from background in regions with clear visual contrast.

3. **Reduced False Positives**: The additional support examples help the model build more robust prototypes, reducing false positive predictions in background regions. The cross-validation across multiple supports improves discrimination.

4. **Higher IoU Scores**: The 5-shot model delivers consistently stronger IoU scores, quantitatively confirming the expected advantage of having more support examples. That said, the overall IoU levels remain modest, suggesting there is still significant headroom for improvement.


**Overall Observations:**

Both models struggle with:
- **Fine Details**: Small-scale structures and thin object parts are frequently missed or poorly segmented
- **Occlusions**: Partially occluded objects present significant challenges
- **Appearance Variation**: Objects with significant scale, pose, or appearance differences from support examples are poorly segmented
- **Complex Backgrounds**: Cluttered or textured backgrounds lead to confusion and false positives

These qualitative results corroborate the quantitative metrics (1-shot: 13.36% best mIoU, 5-shot: 21.67% best mIoU) and highlight the substantial gap to the paper's reported 45.6% performance, suggesting the need for improvements in feature alignment, iterative refinement, or training strategies.

#### Training and Validation Curves

![Training Curves](images/training_curves.png)

*Training loss (left) and validation mIoU (right) for both 1-shot and 5-shot settings over 70 epochs. Training shows steady convergence with 5-shot achieving lower final loss (0.1643 vs 0.2482) and higher validation performance (best mIoU 21.67% vs 13.36%).*

#### Paper Results Comparison

![COCO-20i Paper Results](images/coco-20i_paper_results.png)

*COCO-20^i results reported in the original paper*

![PASCAL-5i Paper Results](images/pascal-5i_paper_results.png)

*PASCAL-5^i results reported in the original paper*

---

# 4. Conclusion

This project implements the ABCB (Addressing Background Context Bias) method for few-shot semantic segmentation as proposed in the CVPR 2024 paper. The implementation includes all three key components: Query Prediction, Support Modulation, and Information Cleansing, with the iterative refinement framework (T=3 iterations).

## Implementation Status

The codebase successfully implements:
- Complete ABCB architecture with ResNet-50/101 backbones
- Iterative refinement mechanism with three-stage processing
- COCO-20^i and PASCAL-5^i dataset loaders with automatic mask generation
- Training pipeline with proper augmentation and optimization
- Evaluation metrics and checkpointing system

## Replication Results

Full training (70 epochs) completed for COCO-20^i with ResNet-50:
- **Completed**: 1-shot and 5-shot settings (single fold each)
- **Computational Cost**: ~80 hours per fold, 160 hours total for both settings
- **Results**:
  - **1-shot**: Final mIoU = 8.84%, Best mIoU = 13.36% (epoch 36)
  - **5-shot**: Final mIoU = 16.61%, Best mIoU = 21.67% (epoch 21)
- **Paper mIoU**: 45.6% (COCO-20^i, ResNet-50, 1-shot)

## Analysis of Performance Gap

Several factors may contribute to the substantial difference between paper results (45.6% mIoU) and current early-training results (near 0%):

### 1. **Performance Gap Despite Full Training**
Despite completing the full 70 epochs with 20,000 training episodes as specified in the paper, the achieved mIoU (13.36% best for 1-shot, 21.67% best for 5-shot) remains significantly below the paper's reported 45.6% for 1-shot. This suggests that training duration alone is not the limiting factor.

### 2. **Backbone Adaptation**
While this implementation keeps the ResNet backbone trainable (assumption not explicitly stated in paper), effective fine-tuning requires many epochs. The frozen vs. trainable backbone decision significantly impacts convergence speed and final performance. The model may need 20-30 epochs just to adapt the backbone features to the few-shot segmentation task.

### 3. **Overfitting in Later Epochs**
Validation mIoU peaks around epoch 21-36 and does not improve further (or slightly degrades). This suggests overfitting where the model memorizes training patterns rather than learning generalizable features. The paper may employ additional regularization techniques or early stopping strategies not documented in the methodology.

### 6. **Hyperparameter Sensitivity**
Several hyperparameters may require tuning for optimal performance:
- Token limits (1024 support, 512 fg, 512 bg) affect feature representation capacity
- Number of attention heads (4) impacts feature alignment granularity
- Batch size (8 for COCO) influences gradient stability and convergence
- Crop size (641×641) determines spatial context availability

Since the project did not include full training across all settings, the optimality of the selected design parameters remains unverified. Additionally, broad hyperparameter searches were not feasible because each fold requires substantial training time (approximately 80 hours) and compute availability was limited. Despite this, the training pipeline is fully implemented for all combinations of dataset (PASCAL-5^i, COCO-20^i), backbone (ResNet-50, ResNet-101), shot count (1, 5), and fold (1–4). The modular implementation facilitates controlled experimentation over batch size, learning rate, crop size, token budgets, attention configuration, and augmentation strategies.



## Conclusion Remarks

This implementation successfully replicates the ABCB architecture with all three key components (Query Prediction, Support Modulation, Information Cleansing) and completes the full 70-epoch training protocol specified in the paper. The codebase provides a complete, modular, and extensible framework for few-shot semantic segmentation research.

**Implementation Achievements:**
- **Complete Architecture**: All components (QP, SM, IC) with iterative refinement (T=3) fully implemented
- **Full Training Protocol**: 70 epochs × 20,000 episodes completed for both 1-shot and 5-shot settings
- **Proper Convergence**: Training loss decreases steadily, validation performance shows learning progression
- **Multi-Configuration Support**: Code supports PASCAL-5^i and COCO-20^i datasets, ResNet-50/101 backbones, all folds
- **Reproducibility**: Fixed seeds, deterministic augmentation, comprehensive logging and checkpointing

**Key Findings:**
- **5-shot Advantage**: 5-shot achieves 21.67% best mIoU vs. 13.36% for 1-shot, demonstrating the expected benefit of additional support examples
- **Learning Dynamics**: Clear learning progression validates architectural correctness
- **Mid-Training Peak**: Best performance at epochs 21-36, followed by overfitting in later epochs
- **Qualitative Results**: Visual examples show the model captures general object locations but struggles with precise boundary delineation. Notably, the model sometimes fails to produce any segmentation mask at all, predicting empty or near-empty outputs for certain examples, indicating instability in the prediction mechanism

**Performance Gap Analysis:**
The achieved mIoU (21.67% best for 5-shot, 13.36% for 1-shot on single fold) remains below the paper's reported 45.6% mean for 1-shot. Potential factors include:
- **Single Fold Limitation**: Results from one fold vs. paper's 4-fold cross-validation mean
- **Implementation Details**: Subtle differences in attention mechanisms, initialization, or regularization not fully specified in the paper
- **Overfitting**: Performance degradation after mid-training suggests need for stronger regularization or early stopping
- **Hyperparameter Space**: Limited exploration due to computational constraints

**Lessons Learned:**
 The training behavior observed—stable convergence, consistently stronger 5-shot performance, and occasional mid-training performance peaks—suggests the implementation is functionally correct. The remaining performance gap emphasizes how critical complete implementation reporting is in research papers, and how difficult it can be to reproduce state-of-the-art results without access to the authors’ code. Finally, the heavy computational cost (over 160 GPU-hours for limited trials) highlights the practical importance of efficient hyperparameter search and the central role of compute availability in modern deep learning research.

---

# 5. References

1. **ABCB Paper**: Zhu, H., et al. (2024). "Addressing Background Context Bias in Few-Shot Segmentation through Iterative Modulation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

2. **PASCAL-5^i Dataset**: Shaban, A., Bansal, S., Liu, Z., Essa, I., & Boots, B. (2017). "One-Shot Learning for Semantic Segmentation." *British Machine Vision Conference (BMVC)*.

3. **COCO-20^i Dataset**: Nguyen, K., & Todorovic, S. (2019). "Feature Weighting and Boosting for Few-Shot Segmentation." *IEEE/CVF International Conference on Computer Vision (ICCV)*.
   - Based on MS-COCO: Lin, T. Y., et al. (2014). "Microsoft COCO: Common Objects in Context." *European Conference on Computer Vision (ECCV)*.

4. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

5. **PASCAL-5^i Implementation Reference**: HSNet - Hypercorrelation Squeeze Network  
   GitHub Repository: [https://github.com/juhongm999/hsnet](https://github.com/juhongm999/hsnet)  
   Min, J., Kang, D., & Cho, M. (2021). "Hypercorrelation Squeeze for Few-Shot Segmentation." *IEEE/CVF International Conference on Computer Vision (ICCV)*.

6. **PyTorch**: Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems (NeurIPS)*.

---

## BibTeX

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@inproceedings{everingham2010pascal,
  title={The PASCAL Visual Object Classes (VOC) Challenge},
  author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
  booktitle={International Journal of Computer Vision},
  volume={88},
  number={2},
  pages={303--338},
  year={2010}
}

@inproceedings{min2021hypercorrelation,
  title={Hypercorrelation Squeeze for Few-Shot Segmentation},
  author={Min, Juhong and Kang, Dahyun and Cho, Minsu},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={6941--6952},
  year={2021}
}
```

---

# Contact

Oguz Kolukısa  
oguzkolukisa1@gmail.com
