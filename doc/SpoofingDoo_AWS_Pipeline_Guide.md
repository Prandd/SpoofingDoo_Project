```markdown
# ☁️ AWS Infrastructure & Data Pipeline Setup Guide (SpoofingDoo Project)
**Document Owner:** Pooh (Infrastructure) & DD (Data Pipeline)
**Current Status:** Phase 3 (Data Splitting on t3.xlarge) | Waiting for AWS vCPU Quota Approval for Phase 4 (GPU)

---

## 🏗️ Architecture Strategy: "EBS Swap" (DevOps Cost Optimization)
เพื่อให้การทำงานกับไฟล์ Dataset ขนาด 117 GB คุ้มค่าที่สุดและไม่เสียค่าเช่า GPU โดยเปล่าประโยชน์ เราใช้กลยุทธ์ **EBS Swap** โดยมองเครื่อง EC2 เป็น "ปลั๊กไฟ" และ EBS Volume เป็น "แฟลชไดรฟ์" ที่สามารถถอดสลับไปเสียบเครื่องไหนก็ได้ใน Availability Zone เดียวกัน (`ap-southeast-1a`)

---

## 🛠️ Phase 1: Storage & Upload Server (complete)
**เป้าหมาย:** สร้างพื้นที่จัดเก็บและเปิดเครื่องสเปคต่ำเพื่อรอรับไฟล์จากเครื่อง Local (Windows) ของ DD

1. **สร้างแฟลชไดรฟ์ (EBS Volume):**
   - Type: `gp3`
   - Size: `500 GiB` (เผื่อพื้นที่แตกไฟล์)
   - IOPS: `3000` | Throughput: `125 MiB/s`
   - Availability Zone: **`ap-southeast-1a`** 
2. **สร้าง Upload Server:**
   - Instance Type: `t3.micro` (save cost approximately around $0.1/hr)
   - Subnet: ล็อกเป้าหมายที่ `ap-southeast-1a`
3. **การตั้งค่าดิสก์ (รันครั้งแรกและครั้งเดียว):**
   ```bash
   lsblk
   sudo mkfs -t ext4 /dev/nvme1n1  # ext4 format
   mkdir ~/dataset
   sudo mount /dev/nvme1n1 ~/dataset
   sudo chown -R ubuntu:ubuntu ~/dataset
   ```
4. **การอัปโหลดไฟล์ (WinSCP):**
   - DD ใช้ WinSCP ล็อกอินด้วย IP และไฟล์ `.pem` (User: `ubuntu`)
   - ลากไฟล์ `CSS_Dataset.tar.gz` (117 GB) ลงโฟลเดอร์ `/home/ubuntu/dataset`

---

## 🔍 Phase 2: Data Verification (complete)
**เป้าหมาย:** ตรวจสอบความสมบูรณ์ของไฟล์หลังการอัปโหลดเพื่อป้องกันไฟล์ Corrupt

1. **เช็คขนาดไฟล์เบื้องต้น:** `ls -lh ~/dataset` (ผลลัพธ์: 118G)
2. **เช็คโครงสร้างไฟล์บีบอัด:** `tar -tvf CSS_Dataset.tar.gz | head -n 10`
3. **เช็คความถูกต้อง (Checksum):**
   - รัน `sha256sum CSS_Dataset.tar.gz` บน AWS เทียบกับคำสั่ง `certutil -hashfile CSS_Dataset.tar.gz SHA256` บนเครื่อง Windows ของ DD

*เมื่อเสร็จสิ้น สั่ง `sudo umount ~/dataset` และ Stop เครื่อง `t3.micro`*

---

## ⚙️ Phase 3: Interim Processing Server (complete)
**เป้าหมาย:** เนื่องจากงาน Stratified Sampling ต้องใช้ RAM สูงในการอ่านไฟล์ `metadata.csv` (เครื่องเดิม RAM 1GB จะค้าง) เราจึงเปิดเครื่อง CPU สเปคกลางมารองรับงานนี้ระหว่างรอ GPU

1. **สร้าง Processing Server:**
   - Instance Type: `t3.xlarge` (4 vCPU, 16 GB RAM)
   - OS Storage (Root volume): ขยายเป็น **`50 GiB`** (ป้องกัน Disk Full ตอนแตกไฟล์/Swap RAM)
   - Subnet: `ap-southeast-1a`
2. **เสียบดิสก์ข้อมูล:**
   - ไปที่เมนู Volumes กด Attach ดิสก์ 500GB ลูกเดิมเข้ากับ `t3.xlarge`
3. **การ Mount ดิสก์เข้าเครื่องใหม่ (ห้าม Format เด็ดขาด):**
   ```bash
   mkdir ~/dataset
   sudo mount /dev/nvme1n1 ~/dataset
   ls -lh ~/dataset  # ยืนยันว่าไฟล์ CSS_Dataset.tar.gz ยังอยู่ครบ
   ```

---

## 💻 งานสำหรับ DD: Strategic Sampling & Splitting
**เป้าหมาย:** สกัดข้อมูล 10% และแบ่ง Train/Val/Test โดยไม่ให้เกิด Data Leakage

**ขั้นตอนการทำงานบนเครื่อง `t3.xlarge`:**
1. **แตกไฟล์:**
   ```bash
   cd ~/dataset
   tar -xzvf CSS_Dataset.tar.gz
   ```
2. **รัน Python Script สำหรับ Data Pipeline:**
   ```python
   import pandas as pd
   import json

   # 1. โหลดข้อมูล
   df = pd.read_csv('metadata.csv')

   # 2. Stratified Balanced Sampling (Bona fide 1 : Spoofed 2 หรือ 1:1)
   bona_fide = df[df['label'] == 'Bona fide']
   spoofed = df[df['label'] == 'Spoofed']

   # สุ่มให้ได้สัดส่วน ~133k ไฟล์ (10%)
   sampled_bona_fide = bona_fide.sample(n=44000, random_state=42)
   sampled_spoofed = spoofed.sample(n=88000, random_state=42)
   sampled_df = pd.concat([sampled_bona_fide, sampled_spoofed])

   # 3. Define Splits (ใช้ speaker_id ป้องกัน Data Leakage)
   train_speakers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   val_speakers = [15, 16, 17]
   test_speakers = [18, 19, 20]

   train_df = sampled_df[sampled_df['speaker_id'].isin(train_speakers)]
   val_df = sampled_df[sampled_df['speaker_id'].isin(val_speakers)]
   test_df = sampled_df[sampled_df['speaker_id'].isin(test_speakers)]

   # 4. Generate Manifest Files
   def create_manifest(dataframe, filename):
       manifest_data = dataframe.to_dict(orient='records')
       with open(filename, 'w') as f:
           json.dump(manifest_data, f, indent=4)

   create_manifest(train_df, 'train_manifest.json')
   create_manifest(val_df, 'val_manifest.json')
   create_manifest(test_df, 'test_manifest.json')
   ```

---

## 🚀 Phase 4: GPU Training Server (Pending AWS Permission status: waiting)
**สถานะปัจจุบัน:** ส่ง Request ขอเพิ่ม Quota `Running On-Demand G and VT instances` เป็น **8 vCPU** เรียบร้อยแล้ว (Case ID: `177696610800047`) รออีเมลอนุมัติจาก AWS

**แผนการเมื่อได้รับการอนุมัติ:**
1. สั่ง Unmount และถอดดิสก์ 500GB ออกจากเครื่อง `t3.xlarge`
2. สร้างเครื่องสำหรับรันโมเดล:
   - **AMI:** `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)`
   - **Instance Type:** **`g4dn.xlarge`** (ประหยัดและสเปคพอเพียง)
   - **Root Volume:** `100 GiB`
   - **Subnet:** `ap-southeast-1a`
3. Attach ดิสก์ 500GB (ที่แตกไฟล์และทำ Manifest เสร็จแล้ว) เข้าเครื่อง GPU
4. สั่ง Mount ดิสก์ และเริ่มขั้นตอน Train Model ได้ทันที!
```