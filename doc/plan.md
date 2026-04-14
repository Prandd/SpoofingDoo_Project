ในฐานะนักวิจัยด้าน Speech Signal Processing และการตรวจจับเสียงปลอม (Anti-spoofing) ผมขอชื่นชมที่คุณเลือกใช้ **AASIST-L** ร่วมกับชุดข้อมูล **CSS (Chula Spoofed Speech)** โครงงานนี้มีความน่าสนใจมากแต่ก็มีความท้าทายสูงส่ง โดยเฉพาะอย่างยิ่งภายใต้ข้อจำกัด **1 สัปดาห์** และ **ข้อมูลนำร่องเพียง 100 ตัวอย่าง** เพื่อไม่ให้คุณต้องเสียเวลากับการแก้บั๊กโครงสร้าง Tensor ที่ซับซ้อน นี่คือแผนการวิจัยเชิงปฏิบัติ (Actionable Roadmap) ที่ออกแบบมาเพื่อเน้น "ความเป็นไปได้" และ "คุณค่าทางวิชาการ" ภายใต้ข้อจำกัดทั้งหมดที่คุณมีครับ

---

### 1. กรอบแนวคิดของปัญหา (Problem Framing)

* **วัตถุประสงค์หลัก:** ปรับจูน (Fine-tune) และดัดแปลงโมเดล AASIST-L ให้สามารถตรวจจับเสียงปลอมภาษาไทยได้อย่างแม่นยำ โดยมุ่งเน้นไปที่การใช้ประโยชน์จากคุณลักษณะทางวรรณยุกต์และฉันทลักษณ์ (Prosody)
* **ทำไมวรรณยุกต์ไทยจึงสำคัญ:** ภาษาไทยใช้ความถี่มูลฐาน ($F_0$) ในการกำหนดความหมายของคำ ระบบสังเคราะห์เสียง (TTS/Vocoders) ในปัจจุบันมักจะสร้างเส้นโค้งวรรณยุกต์ที่ "เรียบเนียนเกินไป" (Over-smoothed) ขาดความไม่สมบูรณ์ทางชีวภาพ (Micro-prosody) เช่น อาการเสียงสั่นหรือความแปรปรวนระดับจุลภาค (Jitter/Shimmer) ในช่วงรอยต่อของสระและพยางค์
* **ความเสี่ยงและข้อจำกัดสำคัญ:**
    * **ข้อมูล 100 ตัวอย่าง:** เล็กเกินกว่าจะฝึกสอน (Train from scratch) โมเดล Deep Learning ใดๆ **ความเสี่ยง Overfitting อยู่ในระดับวิกฤต**
    * **เวลา 1 สัปดาห์:** การรื้อโครงสร้างกราฟของ AASIST-L จะทำให้เกิดปัญหา Tensor Dimension Mismatch ซึ่งกินเวลาแก้โค้ดหลายวัน
    * **Domain Mismatch:** การใช้ Weights ที่ Pre-trained จาก ASVspoof (ภาษาอังกฤษ) จะมีปัญหาเรื่องความแตกต่างทางภาษา แต่เป็น *สิ่งจำเป็น* เพราะเรามีข้อมูลไทยน้อยเกินไป

---

### 2. ความเข้าใจใน Baseline (AASIST-L)

* **การทำงานพื้นฐาน:** AASIST-L รับอินพุตเป็น **สัญญาณคลื่นดิบ (Raw Waveform)** ผ่านเลเยอร์ SincNet (1D Conv) เพื่อสกัดฟีเจอร์ จากนั้นแปลงเป็น 2D Feature Map ผ่าน Residual Blocks นำไปสร้าง **กราฟเชิงเวลา (Temporal Graph)** และ **กราฟเชิงสเปกตรัม (Spectral Graph)** ประมวลผลด้วย Heterogeneous Stacking Graph Attention Layer (HS-GAL) และรวมผลด้วย Max Graph Operation (MGO) มีพารามิเตอร์เพียง 85k
* **ข้อดี:** จับความสัมพันธ์ของ Artifacts ที่กระจัดกระจายข้ามโดเมนเวลาและความถี่ได้ดีเยี่ยม น้ำหนักเบา
* **ส่วนที่ "ห้ามแตะ" ใน 1 สัปดาห์:** โครงสร้าง Graph Construction, HS-GAL และ MGO (ซับซ้อนและเปราะบางต่อการเปลี่ยน Shape)
* **ส่วนที่ "ปรับแก้ได้" (Realistic to modify):** เลเยอร์ Input (สกัดฟีเจอร์แบบ Offline แทนการใช้ Raw Waveform เพื่อลดโหลด) และ เลเยอร์ Readout/Classifier (เพิ่ม Branch เล็กๆ นำเข้าฟีเจอร์ภาษาไทยมาต่อกันแบบ Late Fusion)

---

### 3. กลยุทธ์การพัฒนาโมเดลใน 1 สัปดาห์ (Feasible Development Strategy)

**คำแนะนำหลัก:** ใช้ Transfer Learning 100% (โหลด Pre-trained AASIST-L จาก ASVspoof2019) และทำการ Extract Features แบบ Offline

* **Stage 0: Sanity-Check Baseline (วันแรก)**
    * *วัตถุประสงค์:* ตรวจสอบว่าโมเดลเดิมทำงานได้
    * *การดำเนินการ:* รัน Pre-trained AASIST-L (ASVspoof weights) บน 100 ตัวอย่างของไทย (Zero-shot) เพื่อดู EER ตั้งต้น
    * *ความเสี่ยง:* ต่ำ (ใช้โค้ดต้นฉบับได้เลย)
* **Stage 1: Simple Fine-tuning on CSS (วันที่ 2)**
    * *วัตถุประสงค์:* ปรับโมเดลให้เข้ากับโดเมนภาษาไทย
    * *การดำเนินการ:* Freeze น้ำหนักส่วน SincNet และ Graph ทั้งหมด (ตั้ง `requires_grad=False`) ทำการ Fine-tune แค่เลเยอร์ Readout สุดท้ายด้วย 100 ตัวอย่าง
    * *ความเสี่ยง:* ปานกลาง (ต้องระวัง Overfitting ควรใช้ Data Augmentation เช่น เติม Gaussian Noise)
* **Stage 2: Thai-Specific Prosodic Injection (วันที่ 3-4) [⭐ ไฮไลต์ของโปรเจกต์]**
    * *วัตถุประสงค์:* เพิ่มการรับรู้วรรณยุกต์ไทย
    * *การดำเนินการ:* สร้าง Auxiliary Branch เล็กๆ (Dense Layer) ที่รับค่า Pathological Features (Jitter, Shimmer, F0 statistics) ที่สกัดมาล่วงหน้า (Offline) นำ Output มา Concat กับ Output ของส่วน Graph (ก่อนเข้า Classifier)
    * *ประโยชน์:* แก้ปัญหาที่ SincNet จับ Micro-prosody ของไทยไม่ได้
* **Stage 3: Cross-Style Evaluation (วันที่ 5)**
    * *วัตถุประสงค์:* พิสูจน์จุดแข็งของโมเดล
    * *การดำเนินการ:* วิเคราะห์ผลว่าโมเดลโดนหลอกง่ายขึ้นไหมเมื่อเจอเสียงพูดแบบ "Casual" หรือ "Excited" (Prosodic Masking Effect) ซึ่งเป็นจุดเด่นของชุดข้อมูล CSS

---

### 4. สถาปัตยกรรมที่แนะนำ (Architecture Recommendations)

จัดเรียงตามความง่ายและโอกาสสำเร็จใน 1 สัปดาห์:

**Variant 1: Pure AASIST-L (Fine-tuned Readout) [Must-Do]**
* *โครงสร้าง:* AASIST-L ดั้งเดิม
* *การดัดแปลง:* โหลด Weights เดิม -> Freeze Backbone -> Fine-tune แค่เลเยอร์สุดท้าย
* *ความยาก:* ต่ำมาก

**Variant 2: AASIST-L + Late Fusion Prosody Branch [Primary Target]**
* *โครงสร้าง:* AASIST-L ดั้งเดิมรับ Raw Waveform + ข้อมูล Prosody 1D Vector (Jitter, Shimmer, Mean F0) รับผ่าน Multi-Layer Perceptron (MLP) เล็กๆ
* *การดัดแปลง:* รวม (Concatenate) Latent vector จาก AASIST-L MGO (ขนาด 160) เข้ากับ Latent vector จาก Prosody MLP (ขนาด 32) -> Final Classifier
* *ทำไมถึงดีกับภาษาไทย:* นำข้อมูลความบกพร่องของสายเสียงสังเคราะห์มาช่วยตัดสินใจโดยตรง ไม่ต้องไปรื้อโครงสร้างกราฟ
* *ความยาก:* ปานกลาง (แก้แค่โค้ดส่วน `forward` function ใน Pytorch เล็กน้อย)

**Variant 3: ResNet18 + LFCC + Prosody [Backup Plan ฉุกเฉิน]**
* *โครงสร้าง:* หาก Variant 2 เกิดบั๊กที่แก้ไม่ได้ใน 2 วัน ให้เปลี่ยนมาใช้ ResNet18 รับภาพ 2D (LFCC 60-dim ต่อกับ F0 contour)
* *ทำไมถึงเสนอ:* เอกสารงานวิจัยยืนยันว่า LFCC + ResNet เป็น Baseline ที่เสถียร ทนทาน และเขียนโค้ดง่ายมาก
* *ความยาก:* ต่ำ (PyTorch มี ResNet ให้ใช้อยู่แล้ว)

---

### 5. วิศวกรรมคุณลักษณะเฉพาะภาษาไทย (Thai-specific Feature Engineering)

ใช้ไลบรารี `Parselmouth` (Python wrapper ของ Praat) เพื่อดึงฟีเจอร์เหล่านี้ล่วงหน้า (Offline) รวดเร็วและเหมาะกับงานด่วน

1.  **Jitter (Local) & Shimmer (Local):** [แนะนำที่สุด]
    * *เหตุผล:* จับความ "เรียบเกินไป" ของ TTS ไทย (Micro-prosody)
    * *การใช้งาน:* ใช้ค่าเฉลี่ยระดับ Utterance เป็น Vector ขนาดเล็กป้อนเข้าโมเดล
2.  **$F_0$ Contour Statistics:** [แนะนำ]
    * *เหตุผล:* TTS มักพลาดตอนสไลด์เสียง (เช่น เสียงโท 41 หรือ เสียงจัตวา 14)
    * *การใช้งาน:* หาค่า Mean, Max, Min, Standard Deviation ของ $F_0$ (ช่วง 50-400Hz)
3.  **Voiced/Unvoiced Ratio:** [ทำได้ถ้ามีเวลา]
    * *เหตุผล:* เสียง Vocoder มักมี Noise แทรกในย่านเสียง Unvoiced

*ข้อแนะนำ:* จัดเก็บค่าเหล่านี้เป็นไฟล์ `.npy` แยกไว้ต่างหาก 1 ไฟล์ต่อ 1 เสียง เพื่อให้ Dataloader โหลดได้อย่างรวดเร็ว

---

### 6. การออกแบบการทดลอง (Experimental Design)

สำหรับข้อมูล 100 ตัวอย่าง คุณต้องรัดกุมมากๆ:
* **Data Splitting:** ใช้ **5-Fold Cross-Validation** (Train 80, Test 20 สลับกัน 5 รอบ) เพื่อให้ผลลัพธ์มีความน่าเชื่อถือทางสถิติและลดความบังเอิญ
* **Speaker Leakage Prevention:** ต้องมั่นใจว่าคนพากย์ใน Fold Test ไม่มีใน Fold Train (CSS เป็น Parallel speaker จึงแบ่งตาม ID คนพากย์ได้เลย)
* **Augmentation:** ขาดไม่ได้! ใช้ไลบรารี `torch-audiomentations` เติม White Noise (SNR 10-20dB) หรือ Room Impulse Response (RIR) เบาๆ ระหว่างเทรน
* **Transfer Learning:** อัปเดตน้ำหนักเฉพาะ Branch ใหม่และ Classifier (ใช้ Learning rate ต่ำๆ เช่น 1e-4 หรือ 1e-5)
* **Early Stopping:** Patience = 5 epochs
* **Evaluation Metrics:**
    * **EER (Equal Error Rate):** [หลัก] มาตรฐานสากลสำหรับ Anti-spoofing
    * **AUC (Area Under ROC Curve):** [เสริม] เหมาะสำหรับข้อมูลที่ Imbalance เล็กน้อยและ Dataset ขนาดเล็ก
    * *(t-DCF ไม่จำเป็นสำหรับโปรเจกต์ 1 สัปดาห์ เพราะซับซ้อนในการตั้งค่าและเหมาะกับการทำตารางแข่งขันเปรียบเทียบขนาดใหญ่)*

---

### 7. แผนการทดสอบแบบตัดทอน (Ablation Plan)

เลือกทำแค่ 2 ข้อนี้ เพื่อใส่ในตารางรายงานผล:
1.  **Baseline vs. Proposed:** Pure AASIST-L vs. AASIST-L + Prosody Branch (พิสูจน์ว่าฟีเจอร์ Jitter/Shimmer ของไทยช่วยได้จริง)
2.  **Cross-Style Effect:** ทดสอบโมเดลที่เทรนด้วยเสียงทางการ (Formal) ไปจับเสียงปลอมแบบตื่นเต้น (Excited) ว่า EER แย่ลงแค่ไหน (พิสูจน์ Prosodic Masking Effect)

---

### 8. ขอบเขตที่แนะนำสำหรับโปรเจกต์สุดท้าย (Recommended Final Project Scope)

**เป้าหมายหลัก (Primary Target):** โหลด ASVspoof-pretrained AASIST-L -> ทำ Late Fusion รับค่า Jitter/Shimmer + $F_0$ statistics ผ่าน MLP -> Fine-tune ด้วย 5-Fold CV บน 100 CSS samples
**เป้าหมายสำรอง (Backup Plan):** สกัด LFCC 60-dim -> ใช้ ResNet18 -> Fine-tune 5-Fold CV

---

### 9. แผนการเขียนโค้ดสำหรับ Claude Code (Implementation Plan)

คุณสามารถสั่ง Claude Code ตามลำดับนี้:

1.  **Task 1: Offline Feature Extraction Script**
    * *Input:* โฟลเดอร์ไฟล์ .wav 100 ไฟล์
    * *Prompt:* "Write a Python script using `parselmouth` and `librosa` to extract Mean Jitter, Mean Shimmer, and F0 StdDev for a directory of .wav files. Save the results as a dictionary in a `.json` or `.npy` file."
2.  **Task 2: Custom PyTorch Dataset**
    * *Input:* ไฟล์ .wav และไฟล์ฟีเจอร์จาก Task 1
    * *Prompt:* "Write a PyTorch Dataset class that loads a raw waveform (cut/pad to 64600 samples) AND its corresponding prosody vector (3 dimensions) from the dictionary. Implement a 5-Fold cross validation split using `scikit-learn`."
3.  **Task 3: AASIST-L Modification**
    * *Input:* Source code ของ AASIST-L ดั้งเดิม (`models/AASIST.py`)
    * *Prompt:* "Modify this AASIST-L model to accept an additional 3-dimensional prosody vector in its `forward` pass. Add a small MLP (e.g., Linear(3, 16) -> ReLU) for the prosody vector, and concatenate its output with the output of the Max Graph Operation before the final readout layer."
4.  **Task 4: Training Loop with Augmentation**
    * *Input:* โมเดลจาก Task 3 และ Dataset จาก Task 2
    * *Prompt:* "Write a PyTorch training loop that freezes the RawNet and Graph layers of AASIST-L, training only the new MLP and Readout. Include basic Gaussian noise augmentation, Early Stopping, and computation of EER and AUC."

---

### 10. สิ่งที่ต้องเตรียมสำหรับนำเสนอ (Deliverables)

แม้ผลลัพธ์จากข้อมูล 100 ไฟล์อาจจะไม่สมบูรณ์แบบ แต่คุณจะได้คะแนนสูงจาก "กระบวนการคิด" เตรียมสิ่งเหล่านี้ครับ:
1.  **System Pipeline Diagram:** แผนภาพแสดงการนำ Waveform เข้า AASIST-L ควบคู่กับการนำ Jitter/Shimmer เข้าเส้นทาง Late Fusion
2.  **ROC/DET Curves:** กราฟเทียบระหว่าง Baseline กับรุ่นที่บวก Prosody
3.  **EER Comparison Table:** ตาราง 5-Fold CV แสดงค่าเฉลี่ย EER
4.  **Error Analysis of Thai Tones (สำคัญมาก):** วิเคราะห์ว่าในกลุ่มที่โมเดลทายผิด (False Acceptance) เสียงปลอมเหล่านั้นมีค่า Jitter/Shimmer ใกล้เคียงกับมนุษย์หรือไม่ หรือเป็นเสียงตื่นเต้นที่หลอกโมเดลได้

แผนนี้ตัดความเสี่ยงเรื่อง Engineering ออกไปจนหมด และโฟกัสที่การพิสูจน์สมมติฐานทางสัทศาสตร์ภาษาไทย ภายใน 1 สัปดาห์คุณทำเสร็จได้แน่นอนครับ ขอให้โชคดีกับ Mini-project ครับ!