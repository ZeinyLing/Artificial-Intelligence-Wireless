# Extra Credit: Architectural Innovation  

Extra Credit DA-TCsiNet.ipynb 並包括以下5點
1. DATA Processing from D6
2. CsiNet-LSTM
3. DA-TCsiNet (FULL)
4. Ablation Studies -1 without Doppler-aware Delta Gate
5. Ablation Studies -2 without Last-frame Residual Fusion

資料集部分，因容量太大放置雲端: [Mid-Term data](https://drive.google.com/drive/folders/19CI5paZU2WfIJkCdzJPQuQ1JGN4D3ee9?usp=sharing)

會使用到以下資料:
1. D6_raw.mat
2. D6_temporal/ (此資料夾為D6_raw.mat from DATA Processing from D6) 所以直接下載 可不用跑 DATA Processing from D6
   
只要下載後將 程式中 的各個段落 BASE_DIR 改成 資料夾路徑 即可

## Proposed Method: DA-TCsiNet

## 1. Architecture Design

### Overview
DA-TCsiNet (Doppler-Aware Temporal CsiNet) is proposed to replace CsiNet-LSTM, aiming to:
- Effectively utilize temporal correlation  
- Reduce computational overhead on the UE side  
- Enhance robustness against Doppler effects  

---

### UE / BS Partition

#### UE Side
- **Lightweight CNN Encoder**
  - Input: CSI H(t)
  - Output: compressed vector z(t)
  - Purpose: reduce feedback overhead
- Only encoder is deployed → low complexity

---

#### BS Side
- **Temporal Convolution Network (TCN)**
  - Input sequence: [z(t-T+1), ..., z(t)]
  - Capture multi-scale temporal correlation  

- **Doppler-Aware Gate**
  - Uses: Δz(t) = |z(t) - z(t-1)|
  - Adaptively weights temporal features  

- **Residual Fusion**
  - z_final = z(t) + α z_temporal
  - Preserves latest CSI and stabilizes reconstruction  

- **CNN Decoder**
  - Reconstruct CSI  

---

### Design Highlights
- Temporal modeling shifted from UE to BS  
- LSTM replaced by TCN (parallel computation)  
- Reduced UE-side complexity  

---

## 2. Training Strategy

### Setup
- Sequence length: T = 4  
- Data split: 70 / 15 / 15  
- Epochs: 200  
- Batch size: 32  

### Optimizer
- Adam  
- LR = 1e-3  
- Weight decay = 1e-5  
- Cosine annealing scheduler  

---

### Loss Function
L = MSE + λ · NMSE

---

### Online Adaptation
- Not required  
- Model learns temporal and Doppler dynamics offline  

---

## 3. Ablation Study

### (1) w/o Doppler-Aware Gate
- Remove gating mechanism  
- Keep TCN and residual  

### (2) w/o Residual Fusion
- Remove residual connection  
- Use only temporal features  

---

## 4. Results

### UE Complexity

| Model | UE Params | Total Params | UE Ratio |
|------|----------|-------------|----------|
| CsiNet-LSTM | 69.48M | 137.16M | 50.66% |
| DA-TCsiNet | 67.12M | 145.69M | 46.07% |

---

### Reconstruction Performance

| Model | NMSE (dB) | Corr |
|------|----------|------|
| CsiNet-LSTM | -24.39 | 0.9983 |
| DA-TCsiNet | -27.20 | 0.9992 |

---

### Ablation Results

| Model | Doppler | Residual | NMSE | Corr |
|------|--------|---------|------|------|
| Full | ✓ | ✓ | -27.20 | 0.9992 |
| w/o Doppler | ✗ | ✓ | -27.11 | 0.9992 |
| w/o Residual | ✓ | ✗ | -24.39 | 0.9983 |

---

## 5. Discussion

- Reduced UE-side computation  
- Improved reconstruction accuracy  
- Residual fusion has the largest impact  
- Doppler-aware module is more useful in high mobility scenarios  

---
