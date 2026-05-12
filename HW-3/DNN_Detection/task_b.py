# -*- coding: utf-8 -*-
"""
Task (b) PyTorch version using original utils.py

This version keeps the original OFDM simulation:
    utils.ofdm_simulate()

Only the FC-DNN training/testing part is rewritten in PyTorch.

Task (b):
- QPSK: mu = 2
- SNR = 5, 10, 15, 20, 25 dB
- Pilots = 0, 8, 16, 64
- Metric: BER
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

MU = 6                      # QPSK
N_BITS = K * MU             # 128 bits

# Task (b)
SNR_LIST = [5, 10, 15, 20, 25]
PILOT_LIST = [0, 8, 16, 64]     # 0 = no-pilot

# Important:
# This should match your original main.py config.pred_range.
# For original small FC-DNN, it is often one segment, e.g. 0~15.
# For Task (d) single large DNN, change this to np.arange(0, 128).
PRED_RANGE = np.arange(0, 16)
N_OUTPUT = len(PRED_RANGE)

# Training
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 100000
EPOCHS = 20000
BATCH_SIZE = 512
LR = 1e-3

CP_FLAG = True
CLIPPING_FLAG = False

SAVE_DIR = "./task_b_pytorch_import_utils_results"
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
    """
    Generate pilotCarriers, dataCarriers, pilotValue
    for original utils.ofdm_simulate().
    """

    all_carriers = np.arange(K)

    if P <= 0:
        pilot_carriers = np.array([], dtype=np.int64)
        data_carriers = all_carriers.copy()
        pilot_value = 3 + 3j

    elif P >= K:
        pilot_carriers = all_carriers.copy()
        data_carriers = np.array([], dtype=np.int64)

        # In utils.py, if P >= K:
        #     OFDM_data = pilotValue
        # so pilotValue must be a length-K vector.
        pilot_value = np.ones(K, dtype=complex) * (3 + 3j)

    else:
        pilot_carriers = np.linspace(0, K - 1, P, dtype=np.int64)
        pilot_carriers = np.unique(pilot_carriers)
        data_carriers = np.setdiff1d(all_carriers, pilot_carriers)

        # In utils.py, if P < K:
        #     OFDM_data[pilotCarriers] = pilotValue
        # so scalar is okay.
        pilot_value = 3 + 3j

    return pilot_carriers, data_carriers, pilot_value


# =========================================================
# Channel Generation
# =========================================================
def generate_channel_response():
    """
    Generate one Rayleigh fading channel response.
    """
    h = (
        np.random.randn(CHANNEL_LEN)
        + 1j * np.random.randn(CHANNEL_LEN)
    ) / np.sqrt(2 * CHANNEL_LEN)

    return h.astype(np.complex64)


# =========================================================
# Dataset Generation Using Original utils.py
# =========================================================
def generate_dataset(num_samples, snr_db, P):
    """
    Generate dataset by calling original utils.ofdm_simulate().

    X:
        output of utils.ofdm_simulate()
        shape = [num_samples, 256] when K=64

    y:
        selected bits from codeword according to PRED_RANGE
        shape = [num_samples, len(PRED_RANGE)]
    """

    pilot_carriers, data_carriers, pilot_value = make_pilot_config(P)

    X_list = []
    y_list = []

    for _ in range(num_samples):
        # codeword is the target transmitted bits
        codeword = np.random.binomial(n=1, p=0.5, size=(N_BITS,)).astype(np.float32)

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
def normalize_train_test(X_train, X_test):
    """
    Standardize input features using training-set mean/std.
    This usually stabilizes DNN training.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)


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
# Train
# =========================================================
def train_model(X_train, y_train, model_path):
    train_dataset = OFDMDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    model = FCDNN(
        n_input=X_train.shape[1],
        n_output=y_train.shape[1]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_dataset)

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] Loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)

    return model


# =========================================================
# Test / BER
# =========================================================
def test_model(X_test, y_test, model_path):
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
# Main Task (b)
# =========================================================
def main():
    ber_results = {}

    for P in PILOT_LIST:
        ber_results[P] = []

        for snr_db in SNR_LIST:
            print("\n" + "=" * 80)
            print(f"Task (b) | SNR = {snr_db} dB | Pilots = {P}")
            print("=" * 80)

            print("Generating training data...")
            X_train, y_train = generate_dataset(
                num_samples=TRAIN_SAMPLES,
                snr_db=snr_db,
                P=P
            )

            print("Generating testing data...")
            X_test, y_test = generate_dataset(
                num_samples=TEST_SAMPLES,
                snr_db=snr_db,
                P=P
            )

            X_train, X_test = normalize_train_test(X_train, X_test)

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test shape :", X_test.shape)
            print("y_test shape :", y_test.shape)

            model_path = os.path.join(
                SAVE_DIR,
                f"FCDNN_QPSK_SNR{snr_db}_Pilot{P}_out{N_OUTPUT}.pth"
            )

            print("Training model...")
            train_model(X_train, y_train, model_path)

            print("Testing model...")
            ber = test_model(X_test, y_test, model_path)

            ber_results[P].append(ber)

            print(f"Result | SNR = {snr_db} dB | Pilots = {P} | BER = {ber:.8f}")

    # Save txt results
    txt_path = os.path.join(SAVE_DIR, "task_b_ber_results.txt")
    with open(txt_path, "w") as f:
        f.write("Task (b) BER Results\n")
        f.write(f"PRED_RANGE = {PRED_RANGE[0]} to {PRED_RANGE[-1]}\n")
        f.write(f"N_OUTPUT = {N_OUTPUT}\n\n")

        for P in PILOT_LIST:
            label = "No Pilot" if P == 0 else f"Pilot = {P}"
            f.write(label + "\n")

            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"SNR = {snr_db} dB, BER = {ber:.8f}\n")

            f.write("\n")

    print("\nSaved BER results to:", txt_path)

    # Save csv results
    csv_path = os.path.join(SAVE_DIR, "task_b_ber_results.csv")
    with open(csv_path, "w") as f:
        f.write("Pilot,SNR,BER\n")
        for P in PILOT_LIST:
            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"{P},{snr_db},{ber:.10f}\n")

    print("Saved BER CSV to:", csv_path)

    # Plot
    plt.figure(figsize=(7, 5))

    for P in PILOT_LIST:
        label = "No Pilot" if P == 0 else f"Pilot = {P}"

        # Avoid log(0) in semilogy
        plot_ber = [max(v, 1e-8) for v in ber_results[P]]

        plt.semilogy(
            SNR_LIST,
            plot_ber,
            marker="o",
            linewidth=2,
            label=label
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Task (b): BER vs. SNR with Different Number of Pilots")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(SAVE_DIR, "task_b_BER_vs_SNR.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print("Saved BER figure to:", fig_path)


if __name__ == "__main__":
    main()