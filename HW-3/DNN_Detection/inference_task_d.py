# -*- coding: utf-8 -*-
"""
Task (d) Inference Only Version

This script:
- Loads trained Task (d) single large DNN checkpoints
- Generates testing OFDM samples
- Computes BER
- Saves CSV/TXT results
- Replots BER vs. SNR

Important:
This inference script must use the same model architecture and settings
as the Task (d) training script.
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
# Override utils.Modulation for QPSK
# =========================================================
def Modulation(bits, mu):
    """
    Task (d) uses QPSK only.
    """
    bit_r = bits.reshape((int(len(bits) / mu), mu))

    if mu == 2:
        real = 2 * bit_r[:, 0] - 1
        imag = 2 * bit_r[:, 1] - 1
        return (real + 1j * imag) / np.sqrt(2)

    else:
        raise ValueError("Task (d) inference uses QPSK only, so mu must be 2.")


utils.Modulation = Modulation


# =========================================================
# Basic Config
# =========================================================
SEED = 42

K = 64
CP = 16
CHANNEL_LEN = 16

MU = 2
N_BITS = K * MU             # 128 bits

PRED_RANGE = np.arange(0, 128)
N_OUTPUT = len(PRED_RANGE)  # 128

SNR_LIST = [5, 10, 15, 20, 25]
PILOT_LIST = [0, 8, 16, 64]

TEST_SAMPLES = 100000
BATCH_SIZE = 512

CP_FLAG = True
CLIPPING_FLAG = False

# This should be the folder that contains your trained .pth files
CKPT_DIR = "./task_d_single_large_dnn_results"

# Output folder for inference results
SAVE_DIR = "./task_d_inference_results"
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Normalization
# =========================================================
def normalize_test_only(X_test):
    """
    In the training code, X_train mean/std were used.
    If you did not save train mean/std, this inference version normalizes
    the test set by its own statistics.

    For stricter inference, save mean/std during training and load them here.
    """
    mean = X_test.mean(axis=0, keepdims=True)
    std = X_test.std(axis=0, keepdims=True) + 1e-8
    X_test_norm = (X_test - mean) / std

    return X_test_norm.astype(np.float32)


# =========================================================
# PyTorch Dataset
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
# Large FC-DNN Model
# Must match training version exactly
# =========================================================
class LargeFCDNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(LargeFCDNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, n_output)
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

    model = LargeFCDNN(
        n_input=X_test.shape[1],
        n_output=y_test.shape[1]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
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
    ber_results = {}

    for P in PILOT_LIST:
        ber_results[P] = []

        for snr_db in SNR_LIST:
            print("\n" + "=" * 80)
            print(f"Inference | Task (d) | QPSK | SNR = {snr_db} dB | Pilots = {P}")
            print("=" * 80)

            model_path = os.path.join(
                CKPT_DIR,
                f"LargeDNN_QPSK_SNR{snr_db}_Pilot{P}_out{N_OUTPUT}.pth"
            )

            if not os.path.exists(model_path):
                print("Checkpoint not found:", model_path)
                ber_results[P].append(np.nan)
                continue

            print("Generating test data...")
            X_test, y_test = generate_dataset(
                num_samples=TEST_SAMPLES,
                snr_db=snr_db,
                P=P
            )

            X_test = normalize_test_only(X_test)

            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
            print("Loading checkpoint:", model_path)

            ber = inference_ber(X_test, y_test, model_path)
            ber_results[P].append(ber)

            print(
                f"Result | Task (d) Inference | QPSK | "
                f"SNR = {snr_db} dB | Pilots = {P} | BER = {ber:.8f}"
            )

    # Save TXT
    txt_path = os.path.join(SAVE_DIR, "task_d_inference_ber_results.txt")

    with open(txt_path, "w") as f:
        f.write("Task (d) Inference BER Results\n")
        f.write(f"MU = {MU}\n")
        f.write(f"N_BITS = {N_BITS}\n")
        f.write(f"PRED_RANGE = {PRED_RANGE[0]} to {PRED_RANGE[-1]}\n")
        f.write(f"N_OUTPUT = {N_OUTPUT}\n\n")

        for P in PILOT_LIST:
            label = "No Pilot" if P == 0 else f"Pilot = {P}"
            f.write(label + "\n")

            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"SNR = {snr_db} dB, BER = {ber}\n")

            f.write("\n")

    print("\nSaved TXT results to:", txt_path)

    # Save CSV
    csv_path = os.path.join(SAVE_DIR, "task_d_inference_ber_results.csv")

    with open(csv_path, "w") as f:
        f.write("Pilot,SNR,BER\n")
        for P in PILOT_LIST:
            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"{P},{snr_db},{ber}\n")

    print("Saved CSV results to:", csv_path)

    # Plot
    plt.figure(figsize=(7, 5))

    for P in PILOT_LIST:
        label = "No Pilot" if P == 0 else f"Pilot = {P}"

        plot_ber = []
        for v in ber_results[P]:
            if np.isnan(v):
                plot_ber.append(np.nan)
            else:
                plot_ber.append(max(v, 1e-8))

        plt.semilogy(
            SNR_LIST,
            plot_ber,
            marker="o",
            linewidth=2,
            label=label
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Task (d): Single Large DNN Inference BER vs. SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(SAVE_DIR, "task_d_inference_BER_vs_SNR.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print("Saved BER figure to:", fig_path)


if __name__ == "__main__":
    main()