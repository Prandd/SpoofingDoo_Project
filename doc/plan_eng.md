As a researcher in speech signal processing and spoofing detection (anti-spoofing), I commend your choice to use **AASIST-L** together with the **CSS (Chula Spoofed Speech)** dataset. This project is very interesting but also highly challenging—especially under the constraints of **one week** and **only 100 pilot samples**. To save you from losing time to tricky tensor-shape bugs, here is a practical, actionable research roadmap designed to emphasize **feasibility** and **academic value** under all the constraints you have.

---

### 1. Problem framing

* **Main objective:** Fine-tune and adapt the AASIST-L model so it can accurately detect spoofed Thai speech, with emphasis on exploiting **prosodic** cues (tone and intonation patterns).
* **Why Thai prosody matters:** Thai uses fundamental frequency ($F_0$) to distinguish word meaning. Current speech synthesis (TTS/vocoders) often produces prosody contours that are **over-smoothed**, lacking natural **micro-prosody**—for example voice tremor or fine-scale level variation (**jitter/shimmer**) at vowel and syllable boundaries.
* **Key risks and constraints:**
    * **100 samples:** Too small to **train from scratch** any deep learning model. **Overfitting risk is critical.**
    * **One week:** Reworking AASIST-L’s graph structure tends to cause tensor dimension mismatches that can consume many days of debugging.
    * **Domain mismatch:** Using weights **pre-trained on ASVspoof (English)** introduces language-domain gaps, but it is **necessary** because Thai data is extremely limited.

---

### 2. Understanding the baseline (AASIST-L)

* **Core behavior:** AASIST-L takes **raw waveform** input, uses a SincNet layer (1D conv) for feature extraction, then forms 2D feature maps through residual blocks, builds a **temporal graph** and a **spectral graph**, processes them with a Heterogeneous Stacking Graph Attention Layer (HS-GAL), and aggregates with Max Graph Operation (MGO). It has only about **85k** parameters.
* **Strengths:** Captures time–frequency distributed artifacts very well; lightweight.
* **What not to touch in one week:** Graph construction, HS-GAL, and MGO (complex and fragile to shape changes).
* **What is realistic to modify:** The **input** layer (offline feature extraction instead of raw waveform to reduce load) and the **readout/classifier** (e.g. a small extra branch for Thai-specific features via **late fusion**).

---

### 3. One-week feasible development strategy

**Main recommendation:** Use **transfer learning throughout** (load AASIST-L pre-trained on ASVspoof 2019) and perform **offline** feature extraction.

* **Stage 0: Sanity-check baseline (Day 1)**
    * *Goal:* Verify the original model runs end-to-end.
    * *Action:* Run pre-trained AASIST-L (ASVspoof weights) on your 100 Thai samples (**zero-shot**) and record initial EER.
    * *Risk:* Low (use upstream code as-is).
* **Stage 1: Simple fine-tuning on CSS (Day 2)**
    * *Goal:* Adapt the model to the Thai domain.
    * *Action:* **Freeze** SincNet and all graph weights (`requires_grad=False`); fine-tune **only** the final readout on 100 samples.
    * *Risk:* Medium (watch overfitting; use augmentation, e.g. Gaussian noise).
* **Stage 2: Thai-specific prosodic injection (Days 3–4) [⭐ project highlight]**
    * *Goal:* Improve sensitivity to Thai prosody.
    * *Action:* Add a small **auxiliary branch** (dense layers) fed with **pathological** features (**jitter, shimmer, $F_0$ statistics**) extracted **offline**; concatenate its output with the graph output **before** the classifier.
    * *Benefit:* Mitigates SincNet’s weakness on Thai micro-prosody.
* **Stage 3: Cross-style evaluation (Day 5)**
    * *Goal:* Demonstrate where the model is strong or weak.
    * *Action:* Analyze whether the model is easier to fool on **“casual”** or **“excited”** speech (**prosodic masking**), a strength of the CSS dataset.

---

### 4. Recommended architectures

Ordered by ease and likelihood of success within one week:

**Variant 1: Pure AASIST-L (fine-tuned readout) [must-do]**
* *Structure:* Original AASIST-L.
* *Change:* Load original weights → freeze backbone → fine-tune **only** the last layer.
* *Difficulty:* Very low.

**Variant 2: AASIST-L + late-fusion prosody branch [primary target]**
* *Structure:* Original AASIST-L on raw waveform + a 1D **prosody** vector (jitter, shimmer, mean $F_0$) through a small MLP.
* *Change:* **Concatenate** the latent from AASIST-L MGO (dim 160) with the latent from the prosody MLP (e.g. dim 32) → final classifier.
* *Why it suits Thai:* Uses synthetic-voice **defect** cues directly without rewriting the graph stack.
* *Difficulty:* Medium (small edits to the PyTorch `forward` only).

**Variant 3: ResNet18 + LFCC + prosody [emergency backup]**
* *Structure:* If Variant 2 blocks you for more than ~2 days, switch to ResNet18 on a 2D input (e.g. 60-dim LFCC stacked with $F_0$ contour).
* *Why suggested:* LFCC + ResNet is a **stable**, robust baseline that is easy to code.
* *Difficulty:* Low (ResNet is built into PyTorch).

---

### 5. Thai-specific feature engineering

Use **`parselmouth`** (Python wrapper for Praat) to extract these features **offline**—fast and suitable for a tight timeline.

1. **Local jitter & local shimmer** [highest priority]
    * *Rationale:* Captures **over-smoothed** Thai TTS (micro-prosody).
    * *Usage:* Utterance-level means as a small vector fed to the model.
2. **$F_0$ contour statistics** [recommended]
    * *Rationale:* TTS often fails on **tone glides** (e.g. mid tone 41 or low tone 14).
    * *Usage:* Mean, max, min, std of $F_0$ (e.g. 50–400 Hz band).
3. **Voiced/unvoiced ratio** [if time permits]
    * *Rationale:* Vocoders often inject noise in **unvoiced** regions.

*Tip:* Store vectors as separate **`.npy`** files (one per utterance) so the dataloader stays fast.

---

### 6. Experimental design

With only 100 samples, the protocol must be tight:

* **Data splitting:** Use **5-fold cross-validation** (train 80 / test 20, rotated five times) for more reliable statistics and less luck dependence.
* **Speaker leakage:** Ensure **no test-speaker overlap** with training in any fold (CSS uses parallel speakers—split by speaker ID).
* **Augmentation:** Essential. Use **`torch-audiomentations`**: white noise (SNR 10–20 dB) and/or light **RIR** during training.
* **Transfer learning:** Update mainly the **new branch** and classifier; use a **low** learning rate (e.g. $10^{-4}$ or $10^{-5}$).
* **Early stopping:** patience = 5 epochs.
* **Metrics:**
    * **EER (equal error rate):** [primary] standard for anti-spoofing.
    * **AUC:** [secondary] useful for mild imbalance and small $N$.
    * *(t-DCF is optional for a one-week project—it is fiddly to configure and aimed at large leaderboard-style comparisons.)*

---

### 7. Minimal ablation plan

Do only these two for the results table:

1. **Baseline vs. proposed:** Pure AASIST-L vs. AASIST-L + prosody branch (show that Thai jitter/shimmer features help).
2. **Cross-style effect:** Train on **formal** speech, test spoof in **excited** style—how much does EER degrade? (prosodic masking).

---

### 8. Recommended final project scope

**Primary target:** Load ASVspoof-pretrained AASIST-L → late fusion of jitter/shimmer + $F_0$ stats via MLP → fine-tune with **5-fold CV** on 100 CSS samples.

**Backup:** Extract 60-dim LFCC → ResNet18 → fine-tune with 5-fold CV.

---

### 9. Implementation plan (for coding assistants)

You can drive implementation in this order:

1. **Task 1: Offline feature extraction script**
    * *Input:* Folder of ~100 `.wav` files.
    * *Prompt:* “Write a Python script using `parselmouth` and `librosa` to extract mean jitter, mean shimmer, and F0 standard deviation for a directory of `.wav` files. Save the results as a dictionary in a `.json` or `.npy` file.”
2. **Task 2: Custom PyTorch dataset**
    * *Input:* `.wav` files and feature file from Task 1.
    * *Prompt:* “Write a PyTorch `Dataset` that loads a raw waveform (cut/pad to 64600 samples) **and** its matching 3-D prosody vector from the dictionary. Implement 5-fold cross-validation splits with `scikit-learn`.”
3. **Task 3: AASIST-L modification**
    * *Input:* Original AASIST-L source (e.g. `models/AASIST.py`).
    * *Prompt:* “Modify this AASIST-L model so `forward` accepts an extra 3-D prosody vector. Add a small MLP (e.g. `Linear(3, 16)` → ReLU), concatenate its output with the MGO output before the final readout.”
4. **Task 4: Training loop with augmentation**
    * *Input:* Model from Task 3, dataset from Task 2.
    * *Prompt:* “Write a PyTorch training loop that freezes the RawNet and graph parts of AASIST-L and trains only the new MLP and readout. Include Gaussian noise augmentation, early stopping, and EER + AUC.”

---

### 10. Deliverables for presentation

Even if metrics on 100 files are imperfect, you can score highly on **clear reasoning**. Prepare:

1. **System pipeline diagram:** Waveform → AASIST-L alongside jitter/shimmer → late fusion path.
2. **ROC/DET curves:** Baseline vs. prosody-augmented model.
3. **EER comparison table:** 5-fold CV mean EER.
4. **Error analysis on Thai tones (very important):** For **false accepts**, do spoofs have jitter/shimmer near human ranges, or does **excited** prosody fool the model?

This plan trims engineering risk and focuses on testing **Thai phonetics/prosody hypotheses**—something you can realistically finish in one week. Good luck with the mini-project!
