# Exercise 2.15: CSI Compression and Reconstruction using CsiNet and CS-CsiNet

## Dataset Preparation and Directory Structure

This exercise uses the COST2100 channel model to generate six different CSI datasets for evaluating CsiNet and CS-CsiNet. First, download the COST2100 official repository from:

```text
https://github.com/cost2100/cost2100
```

After downloading the repository, place the MATLAB scripts `generate_D1_to_D6_raw.m` and `convert_D1_to_D6_to_csinet.m` under the `cost2100-master/` directory.

---

## Dataset Settings

**Table 1. Dataset Settings under Different User Distributions and Channel Scenarios**

| Dataset | Network | Scenario | User distribution |
|---|---|---|---|
| D1 | `Indoor_CloselySpacedUser_2_6GHz` | LOS | Closely-spaced users |
| D2 | `Indoor_CloselySpacedUser_2_6GHz` | LOS | Center-clustered |
| D3 | `Indoor_CloselySpacedUser_2_6GHz` | LOS | Spread / edge users |
| D4 | `SemiUrban_CloselySpacedUser_2_6GHz` | LOS | Closely-spaced users |
| D5 | `SemiUrban_CloselySpacedUser_2_6GHz` | LOS | Well-separated users |
| D6 | `SemiUrban_CloselySpacedUser_2_6GHz` | NLOS | Well-separated users |

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

## CsiNet and CS-CsiNet Performance Comparison

**Table 2. Comparison Results between CsiNet and CS-CsiNet**

| Dataset | CsiNet NMSE dB | CsiNet Corr. | CS-CsiNet NMSE dB | CS-CsiNet Corr. |
|---|---:|---:|---:|---:|
| D1 | -14.5340 | 0.1563 | **-15.3053** | 0.1502 |
| D2 | -13.2373 | 0.1355 | **-14.8795** | **0.1366** |
| D3 | **-13.3436** | 0.1293 | -13.1215 | **0.1296** |
| D4 | **-17.3088** | 0.1120 | -16.5968 | **0.1269** |
| D5 | -10.9518 | 0.1634 | **-12.6152** | **0.1638** |
| D6 | -8.6231 | 0.1602 | **-9.9710** | **0.1604** |

**Note:** NMSE dB is lower-is-better, while correlation is higher-is-better.

---

## Single-Domain vs Mixed-Dataset CsiNet Training

**Table 3. Comparison Results between Single-Domain and Mixed-Trained CsiNet**

| Dataset | Single-domain CsiNet NMSE dB | Mixed-trained CsiNet NMSE dB | Single Corr. | Mixed Corr. |
|---|---:|---:|---:|---:|
| D1 | -14.5340 | **-19.1995** | **0.1563** | 0.1503 |
| D2 | -13.2373 | **-18.3595** | **0.1355** | 0.1342 |
| D3 | -13.3436 | **-19.4212** | **0.1293** | 0.1273 |
| D4 | -17.3088 | **-19.6350** | **0.1120** | 0.1107 |
| D5 | -10.9518 | **-15.3630** | **0.1634** | 0.1606 |
| D6 | -8.6231 | **-12.0599** | 0.1602 | **0.1637** |

Mixed-trained CsiNet achieves lower NMSE on all six datasets, showing that mixed-dataset training improves CSI reconstruction accuracy and generalization across different channel scenarios.

---

## Single-Domain vs Mixed-Dataset CS-CsiNet Training

**Table 4. Comparison Results between Single-Domain and Mixed-Trained CS-CsiNet**

| Dataset | Single-domain CS-CsiNet NMSE dB | Mixed-trained CS-CsiNet NMSE dB | Single Corr. | Mixed Corr. |
|---|---:|---:|---:|---:|
| D1 | -15.3053 | **-18.4032** | 0.1502 | **0.1519** |
| D2 | -14.8795 | **-18.3971** | 0.1366 | **0.1372** |
| D3 | -13.1215 | **-17.4947** | 0.1296 | **0.1297** |
| D4 | -16.5968 | **-16.9633** | **0.1269** | 0.1127 |
| D5 | -12.6152 | **-15.3099** | **0.1638** | 0.1605 |
| D6 | -9.9710 | **-13.2906** | 0.1604 | **0.1688** |

Mixed-trained CS-CsiNet also improves NMSE on all datasets. This indicates that training on diverse channel distributions helps the decoder learn more general CSI reconstruction patterns, even when the encoder is a fixed random projection.

---

