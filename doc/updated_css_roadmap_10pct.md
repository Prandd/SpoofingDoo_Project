# Research Roadmap: Thai Anti-Spoofing with AASIST-L (10% CSS Scale)

## 1. Project Overview
- **Objective:** Fine-tune the AASIST-L model to detect Thai spoofed speech using 10% of the Chula Spoofed Speech (CSS) dataset.
- **Scale:** 133,210 utterances (Approx. 160 hours of audio).
- **Target Architecture:** AASIST-L (85k parameters) with Transfer Learning from ASVspoof 2019.

## 2. Dataset Strategy (10% Subset)
- **Selection:** 133,210 samples sampled from the full 1.3M CSS dataset.
- **Sampling Method:** Stratified sampling to ensure representation across 20 speakers and all styles (Formal, Casual, Excited).
- **Split Protocol:** - **Train (70%) / Val (15%) / Test (15%)**.
    - **Rule:** Strict Speaker-Disjoint split (Speakers in Test set must never appear in Training).
- **Audio Processing:** All raw waveforms must be resampled to 16kHz and cropped/padded to exactly 64,600 samples (approx. 4 seconds).

## 3. Optimized Data Pipeline
Given the scale of 133k samples, efficiency is paramount:
- **Parallel Feature Extraction:** Utilize `joblib` or `torch.multiprocessing` to extract Jitter/Shimmer and F0 stats offline using `parselmouth`. Store as `.npy` for fast I/O.
- **Dataloader Configuration:** - `num_workers`: Set to the number of CPU cores (e.g., 4 or 8).
    - `pin_memory=True`: Speed up tensor transfer to GPU.
    - `prefetch_factor`: Load next batches while the current one is training.
- **Data Augmentation:** Apply on-the-fly augmentation (Gaussian noise, RIR) to increase model robustness against environmental variance.

## 4. ML Modeling Decisions
- **Training Strategy:** Full Fine-tuning. Unlike the pilot study, 133k samples allow unfreezing the entire AASIST-L architecture.
- **Learning Rate Policy:** - Use **Discriminative Learning Rates**: Lower LR (e.g., 1e-6) for the SincNet/Graph backbone; Higher LR (e.g., 1e-4) for the readout/classification head.
    - Implement **Cosine Annealing** learning rate scheduler for better convergence.
- **Late Fusion:** Integrate the 3-D prosody vector (Jitter, Shimmer, F0 std) via a small MLP (Linear-ReLU-Linear) before the final classification layer.

## 5. Cloud GPU Selection & Infrastructure
Recommended cloud resources for training within a one-week timeline:

| GPU Provider | Instance Type | GPU Model | VRAM | Est. Cost/Hour | Why Choose? |
|--------------|---------------|-----------|------|----------------|-------------|
| **Lambda Labs** | 1x A10 | NVIDIA A10 | 24GB | ~$0.60 | Best price-to-performance for AASIST-L. |
| **AWS** | g5.xlarge | NVIDIA A10G | 24GB | ~$1.00 | Highly reliable, integrated with S3 for data storage. |
| **GCP** | g2-standard-4 | NVIDIA L4 | 24GB | ~$0.70 | Efficient and modern architecture. |
| **AWS (Budget)**| g4dn.xlarge | NVIDIA T4 | 16GB | ~$0.52 | Slowest but cheapest; good for debugging. |

### Decision Criteria:
1. **VRAM (Critical):** AASIST-L needs ~4-8GB for moderate batch sizes. 16GB-24GB is ideal for large batch sizes (e.g., 32 or 64) to speed up training.
2. **Cost vs. Time:** A10/A10G is the "Sweet Spot". It is ~3x faster than T4 for only ~2x the price.
3. **Storage Speed:** Ensure the instance uses NVMe SSDs. HDD storage will bottleneck the training as the script spends time waiting for 133k audio files to load.

## 6. Evaluation Metrics
- **Primary:** Equal Error Rate (EER).
- **Secondary:** Area Under Curve (AUC).
- **Ablation:** Compare performance on "Formal" vs "Excited" speech styles to evaluate prosodic masking effects in Thai language.
