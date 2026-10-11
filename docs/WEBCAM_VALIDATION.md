# Webcam Domain Validation Protocol — EyeHealthAI

**Document**: `docs/WEBCAM_VALIDATION.md`  
**Purpose**: Specification and Clinical Study Protocol for Future Prospective Laptop-Webcam Validation  
**Status**: Layer 6 Specification (Pre-Deployment Clinical Protocol)  
**Date**: 2026-10-11  

---

## 1. Background & Optical Domain Discrepancy

The EyeHealthAI Anemia Model A (`AnemiaRegressor`, ResNet-50) was trained and validated on clinical research datasets (`India.xlsx` and `Italy.xlsx`). The original captures were acquired using handheld digital cameras or smartphones positioned at macro distances with relatively stable exposure.

Ordinary consumer laptop webcams present severe optical and signal-processing confounders:

1. **Hardware & Sensor Limitations**:
   - **Low Signal-to-Noise Ratio (SNR)**: Tiny CMOS sensor dies (often 1/5" or 1/6") with small pixel pitch (1.0–1.4 µm) exhibit high photon shot noise in typical indoor lighting.
   - **Fixed Focal Distance**: Fixed at infinity or 50–70 cm (monitor-to-face distance), resulting in severe defocus blur when an eye is brought close (15–20 cm) for conjunctival inspection.
2. **Proprietary Image Signal Processor (ISP) Pipelines**:
   - **Auto-White-Balance (AWB)**: ISPs dynamically adjust color temperature and tint to neutralize ambient illumination. This directly alters conjunctival erythema and pallor chromaticity, which is the exact biological signal used for Hb estimation.
   - **Dynamic Range Compression & Tone Mapping**: Non-linear local contrast enhancements and skin-smoothing filters degrade fine vascular textures.
3. **Clinical Domain Gate Rationale**:
   - In accordance with `AGENTS.md` rules on data integrity and clinical safety, webcam images **cannot be silently fed into clinical models** or presented as confirmed screening results.
   - By default, webcam inference is assigned the status `blocked_domain_validation`.

---

## 2. Prospective Study Design & Clinical Protocol

To validate or adapt the model for laptop webcam use, a prospective validation study must follow this protocol:

### 2.1 Cohort & Sampling Plan
- **Sample Size**: Minimum $N = 150$ unique human subjects.
- **Diversity**: Stratified representation across:
  - Age (18–80 years).
  - Biological sex (50% male, 50% female).
  - Fitzpatrick skin phototypes I through VI.
  - Hemoglobin distribution: balanced across severe anemia ($<8.0$ g/dL), moderate anemia ($8.0–10.9$ g/dL), mild anemia ($11.0–11.9$ g/dL for females, $11.0–12.9$ g/dL for males), and non-anemic controls.
- **Ethical Review**: Prior approval from an institutional review board (IRB) with signed informed consent.

### 2.2 Gold-Standard Clinical Ground Truth
- Within $\le 15$ minutes of webcam image acquisition, a certified phlebotomist must obtain a venous blood sample.
- Ground-truth hemoglobin (Hb in g/dL) must be measured using a certified clinical laboratory automated hematology analyzer (e.g. Sysmex XN-series, Beckman Coulter DxH, or Abbott Alinity).
- Fingerstick point-of-care devices (e.g. HemoCue 201+) may serve as secondary references, but automated venous counters are required for primary benchmarking.

### 2.3 Hardware Matrix (Cross-Device Validation)
Captures must be replicated across three standard consumer laptop camera tiers:
1. **Tier 1 (High-volume 720p)**: Standard integrated laptop webcam (e.g., Realtek/Sunplus 720p @ 30 fps).
2. **Tier 2 (Modern 1080p)**: Full HD laptop webcam (e.g., Windows Hello FHD 1080p).
3. **Tier 3 (Apple Silicon FaceTime HD)**: 1080p sensor with Apple M-series neural ISP.

---

## 3. Standardization & Capture Guidance Protocol

During prospective acquisition, the capture environment must be standardized:

### 3.1 Illumination Standard
- Fixed 5500K diffuse illumination source (CRI $\ge 95$) positioned directly above or behind the laptop display.
- Elimination of mixed illumination (e.g. fluorescent ceiling lights mixed with incandescent warm lamps).
- Ambient illuminance maintained between 400 and 600 lux at the ocular plane.

### 3.2 Patient Positioning & Framing
- Patient seated comfortably, 15–20 cm from the laptop camera.
- Alignment within the dashboard's circular/oval target guide.
- Trained operator performs gentle manual eversion of the lower eyelid while patient directs gaze upward to expose the inferior palpebral and forniceal conjunctiva.

### 3.3 Live Image Quality Guidance
Frames must satisfy objective pre-screening criteria before capture:
- **Sharpness (Laplacian variance proxy)**: $\ge 40.0$ to ensure vascular structures are resolvable.
- **Luminance (Mean gray intensity)**: $40.0 \le \text{Brightness} \le 210.0$ to prevent clipped underexposure or blooming overexposure.
- **Usability Status**: Must be verified as `"usable"` by `calculate_image_quality`.

---

## 4. Evaluation Metrics & Progression Gates

The webcam modality will only be unlocked from `blocked_domain_validation` to un-gated deployment when the following empirical gates are achieved:

| Metric | Target Threshold | Clinical Significance |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $\le 1.25$ g/dL | Non-invasive optical screening precision |
| **Root Mean Square Error (RMSE)** | $\le 1.75$ g/dL | Penalizes extreme outlier predictions |
| **Pearson Correlation ($r$)** | $\ge 0.70$ | Linear correspondence with laboratory Hb |
| **Bland-Altman Mean Bias** | $\le \pm 0.30$ g/dL | Verifies absence of systematic over/underestimation |
| **Anemia Screening Sensitivity (WHO cutoff)** | $\ge 85.0\%$ | Minimizes false negatives in high-risk patients |
| **Anemia Screening Specificity (WHO cutoff)** | $\ge 80.0\%$ | Limits unnecessary secondary clinical referrals |

---

## 5. Privacy & Data Handling Guarantees

1. **Zero Continuous Video Recording**: Streaming video is rendered strictly in local browser memory (`<video>` element) and discarded.
2. **Explicit User Capture**: Only frames explicitly triggered by the user via "Capture Still" are processed.
3. **In-Memory Volatility**: The backend API processes still frames in memory (`io.BytesIO`) without writing temporary files to persistent disk storage.
4. **Audit Session Reports**: Downloaded session JSON reports contain non-identifying file basenames, quality scores, timestamps, and model outputs without facial biometric records.
