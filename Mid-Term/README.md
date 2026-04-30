# Exercise 2.15: CSI Compression and Reconstruction using CsiNet and CS-CsiNet

## Dataset Preparation and Directory Structure

This exercise uses the COST2100 channel model to generate six different CSI datasets for evaluating CsiNet and CS-CsiNet. First, download the COST2100 official repository from:

```text
https://github.com/cost2100/cost2100
```

After downloading the repository, place the MATLAB scripts `generate_D1_to_D6_raw.m` and `convert_D1_to_D6_to_csinet.m` under the `cost2100-master/` directory.

---

## Step 1: Generate Raw COST2100 Channel Data

Run the following MATLAB script:

```matlab
generate_D1_to_D6_raw
```

This script generates six raw channel datasets:

```text
D1_raw.mat
D2_raw.mat
D3_raw.mat
D4_raw.mat
D5_raw.mat
D6_raw.mat
```

Each raw file contains the original COST2100 channel response, including variables such as `H_transfer`, `H_norm`, user positions, velocities, scenario settings, and channel parameters.

After this step, the directory structure should look like:

```text
.
└── cost2100-master/
    ├── generate_D1_to_D6_raw.m
    ├── convert_D1_to_D6_to_csinet.m
    ├── D1_raw.mat
    ├── D2_raw.mat
    ├── D3_raw.mat
    ├── D4_raw.mat
    ├── D5_raw.mat
    └── D6_raw.mat
```

---

## Step 2: Convert Raw Data to CsiNet Format

Next, run:

```matlab
convert_D1_to_D6_to_csinet
```

This script converts the raw COST2100 channel data into the input format required by CsiNet and CS-CsiNet. The complex CSI matrix is separated into real and imaginary parts, normalized, reshaped into a `32 × 32 × 2` CSI image, and then flattened into a `2048`-dimensional vector.

Each dataset is split into training, validation, and testing sets:

| File Name | Description |
|---|---|
| `DATA_Htrainin.mat` | Training data |
| `DATA_Hvalin.mat` | Validation data |
| `DATA_Htestin.mat` | Testing data |
| `DATA_HtestFin_all.mat` | Original frequency-domain test CSI for correlation evaluation |

After conversion, the directory structure becomes:

```text
.
└── cost2100-master/
    ├── generate_D1_to_D6_raw.m
    ├── convert_D1_to_D6_to_csinet.m
    ├── data_D1/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── data_D2/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── data_D3/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── data_D4/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── data_D5/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── data_D6/
    │   ├── DATA_Htrainin.mat
    │   ├── DATA_Hvalin.mat
    │   ├── DATA_Htestin.mat
    │   └── DATA_HtestFin_all.mat
    ├── D1_raw.mat
    ├── D2_raw.mat
    ├── D3_raw.mat
    ├── D4_raw.mat
    ├── D5_raw.mat
    └── D6_raw.mat
```

---

## Dataset Generation Summary

The full data preparation flow is:

```text
COST2100 official repository
        ↓
generate_D1_to_D6_raw.m
        ↓
D1_raw.mat ~ D6_raw.mat
        ↓
convert_D1_to_D6_to_csinet.m
        ↓
data_D1 ~ data_D6
        ↓
CsiNet / CS-CsiNet training and testing
```

In this experiment, D1–D6 represent different user distributions and channel scenarios. D1–D3 are generated under an indoor LOS environment with different user distributions, while D4–D6 are generated under a semi-urban environment, including LOS and NLOS settings. These datasets are used to evaluate how CsiNet and CS-CsiNet perform under different channel conditions.

---

## 中文說明

本實驗使用 COST2100 channel model 產生六組 CSI datasets，用於評估 CsiNet 與 CS-CsiNet 的 CSI compression and reconstruction performance。首先需要從 COST2100 官方 GitHub repository 下載程式：

```text
https://github.com/cost2100/cost2100
```

下載後，將 `generate_D1_to_D6_raw.m` 與 `convert_D1_to_D6_to_csinet.m` 放入 `cost2100-master/` 目錄下。

### Step 1：產生 COST2100 raw channel data

先在 MATLAB 中執行：

```matlab
generate_D1_to_D6_raw
```

此程式會產生六組 raw channel datasets，分別為：

```text
D1_raw.mat
D2_raw.mat
D3_raw.mat
D4_raw.mat
D5_raw.mat
D6_raw.mat
```

這些 raw files 會儲存 COST2100 產生的原始 channel response，例如 `H_transfer`、`H_norm`、user positions、user velocities、scenario settings 與相關 channel parameters。

### Step 2：轉換為 CsiNet / CS-CsiNet 可使用格式

接著執行：

```matlab
convert_D1_to_D6_to_csinet
```

此程式會將 `D1_raw.mat` 到 `D6_raw.mat` 轉換成 CsiNet 與 CS-CsiNet 所需的資料格式。轉換過程會將 complex CSI matrix 分成 real part 與 imaginary part，進行 normalization，reshape 成 `32 × 32 × 2` 的 CSI image，最後 flatten 成 `2048` 維向量。

每組 dataset 會被切分成：

| File Name | 說明 |
|---|---|
| `DATA_Htrainin.mat` | Training data |
| `DATA_Hvalin.mat` | Validation data |
| `DATA_Htestin.mat` | Testing data |
| `DATA_HtestFin_all.mat` | Frequency-domain test CSI，用於 correlation 評估 |

簡單來說，本實驗的資料產生流程為：

```text
COST2100 official repository
        ↓
generate_D1_to_D6_raw.m
        ↓
D1_raw.mat ~ D6_raw.mat
        ↓
convert_D1_to_D6_to_csinet.m
        ↓
data_D1 ~ data_D6
        ↓
CsiNet / CS-CsiNet training and testing
```

這樣即可將 COST2100 產生的 raw channel data 轉換成 CsiNet 與 CS-CsiNet 可以直接讀取的 `.mat` 資料格式。
