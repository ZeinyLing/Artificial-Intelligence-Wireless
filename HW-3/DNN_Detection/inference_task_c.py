# -*- coding: utf-8 -*-
"""
Inference-only version:
- Load trained FC-DNN models
- Generate testing data
- Compute BER
- Redraw BER vs. SNR figure

This script does NOT retrain models.
"""

from __future__ import division

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils


# =========================================================
# Basic Config
# =========================================================
SEED = 42

K = 64
CP = 16
CHANNEL_LEN = 16

# =========================================================
# IMPORTANT:
# QPSK:    MU = 2
# 64-QAM:  MU = 6
# =========================================================
MU = 6
N_BITS = K * MU

SNR_LIST = [5, 10, 15, 20, 25]
PILOT_LIST = [0, 8, 16, 64]

# Must match the training setting
PRED_RANGE = np.arange(48, 96)
N_OUTPUT = len(PRED_RANGE)

TEST_SAMPLES = 100000
BATCH_SIZE = 512

CP_FLAG = True
CLIPPING_FLAG = False

# Folder containing trained .pth models
MODEL_DIR = "./task_c_64qam_pytorch_results"

# Output folder for inference results
SAVE_DIR = "./task_c_inference_redraw_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# Device
# =========================================================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Pilot Config
# =========================================================
def make_pilot_config(P):
    all_carriers = np.arange(K)

    if P <= 0:
        pilot_carriers = np.array([], dtype=np.int64)
        data_carriers = all_carriers.copy()
        pilot_value = 3 + 3j

    elif P >= K:
        pilot_carriers = all_carriers.copy()
        data_carriers = np.array([], dtype=np.int64)
        pilot_value = np.ones(K, dtype=complex) * (3 + 3j)

    else:
        pilot_carriers = np.linspace(0, K - 1, P, dtype=np.int64)
        pilot_carriers = np.unique(pilot_carriers)
        data_carriers = np.setdiff1d(all_carriers, pilot_carriers)
        pilot_value = 3 + 3j

    return pilot_carriers, data_carriers, pilot_value


# =========================================================
# Channel Generation
# =========================================================
def generate_channel_response():
    h = (
        np.random.randn(CHANNEL_LEN)
        + 1j * np.random.randn(CHANNEL_LEN)
    ) / np.sqrt(2 * CHANNEL_LEN)

    return h.astype(np.complex64)


# =========================================================
# Dataset Generation
# =========================================================
def generate_dataset(num_samples, snr_db, P):
    pilot_carriers, data_carriers, pilot_value = make_pilot_config(P)

    X_list = []
    y_list = []

    for _ in range(num_samples):
        codeword = np.random.binomial(
            n=1,
            p=0.5,
            size=(N_BITS,)
        ).astype(np.float32)

        channel_response = generate_channel_response()

        feature, _ = utils.ofdm_simulate(
            codeword=codeword,
            channelResponse=channel_response,
            SNRdb=snr_db,
            mu=MU,
            CP_flag=CP_FLAG,
            K=K,
            P=P,
            CP=CP,
            pilotValue=pilot_value,
            pilotCarriers=pilot_carriers,
            dataCarriers=data_carriers,
            Clipping_Flag=CLIPPING_FLAG
        )

        X_list.append(feature.astype(np.float32))
        y_list.append(codeword[PRED_RANGE].astype(np.float32))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


# =========================================================
# Dataset
# =========================================================
class OFDMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# FC-DNN Model
# =========================================================
class FCDNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(FCDNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 500),
            nn.ReLU(),

            nn.Linear(500, 250),
            nn.ReLU(),

            nn.Linear(250, 120),
            nn.ReLU(),

            nn.Linear(120, n_output)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Inference / BER
# =========================================================
def inference_ber(X_test, y_test, model_path):
    test_dataset = OFDMDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    model = FCDNN(
        n_input=X_test.shape[1],
        n_output=y_test.shape[1]
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_errors = 0
    total_bits = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = torch.sigmoid(logits)
            pred_bits = (probs > 0.5).float()

            total_errors += (pred_bits != yb).sum().item()
            total_bits += yb.numel()

    ber = total_errors / total_bits
    return ber


# =========================================================
# Main Inference
# =========================================================
def main():
    print("MU =", MU)
    print("N_BITS =", N_BITS)
    print("PRED_RANGE =", PRED_RANGE[0], "to", PRED_RANGE[-1])
    print("N_OUTPUT =", N_OUTPUT)

    ber_results = {}

    for P in PILOT_LIST:
        ber_results[P] = []

        for snr_db in SNR_LIST:
            print("\n" + "=" * 80)
            print(f"Inference | SNR = {snr_db} dB | Pilots = {P}")
            print("=" * 80)

            model_path = os.path.join(
                MODEL_DIR,
                f"FCDNN_64QAM_SNR{snr_db}_Pilot{P}_out{N_OUTPUT}.pth"
            )

            if not os.path.exists(model_path):
                print("Model not found:", model_path)
                ber_results[P].append(np.nan)
                continue

            print("Loading model:", model_path)

            print("Generating testing data...")
            X_test, y_test = generate_dataset(
                num_samples=TEST_SAMPLES,
                snr_db=snr_db,
                P=P
            )

            # =====================================================
            # IMPORTANT:
            # This uses test-set normalization only.
            # Ideally, training mean/std should be saved and reused.
            # If your training did not save mean/std, this is acceptable
            # for rechecking the curve but not perfectly identical.
            # =====================================================
            mean = X_test.mean(axis=0, keepdims=True)
            std = X_test.std(axis=0, keepdims=True) + 1e-8
            X_test = ((X_test - mean) / std).astype(np.float32)

            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)

            ber = inference_ber(X_test, y_test, model_path)
            ber_results[P].append(ber)

            print(f"BER = {ber:.8f}")

    # =========================================================
    # Save CSV
    # =========================================================
    csv_path = os.path.join(SAVE_DIR, "inference_ber_results.csv")

    with open(csv_path, "w") as f:
        f.write("Pilot,SNR,BER\n")

        for P in PILOT_LIST:
            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"{P},{snr_db},{ber:.10f}\n")

    print("\nSaved CSV to:", csv_path)

    # =========================================================
    # Plot
    # =========================================================
    plt.figure(figsize=(7, 5))

    for P in PILOT_LIST:
        label = "No Pilot" if P == 0 else f"Pilot = {P}"

        plot_ber = []
        valid_snr = []

        for snr_db, ber in zip(SNR_LIST, ber_results[P]):
            if not np.isnan(ber):
                valid_snr.append(snr_db)
                plot_ber.append(max(ber, 1e-8))

        plt.semilogy(
            valid_snr,
            plot_ber,
            marker="o",
            linewidth=2,
            label=label
        )

    title_qam = "QPSK" if MU == 2 else "64-QAM" if MU == 6 else f"{2 ** MU}-QAM"

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"Inference Only: {title_qam} BER vs. SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(SAVE_DIR, "inference_BER_vs_SNR.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print("Saved figure to:", fig_path)


if __name__ == "__main__":
    main()