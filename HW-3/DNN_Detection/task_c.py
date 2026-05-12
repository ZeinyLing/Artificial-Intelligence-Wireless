# -*- coding: utf-8 -*-
"""
Task (c) PyTorch Version Using Original utils.py

Task (c):
- Change mu = 6 for 64-QAM
- Change pred_range = np.arange(48, 96)
- Change network output size to n_output = 48
- Use original utils.ofdm_simulate()
- Override utils.Modulation() to correctly support 64-QAM
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
# Override utils.Modulation for QPSK / 64-QAM
# =========================================================
def Modulation(bits, mu):
    """
    Support:
        mu = 2  -> QPSK
        mu = 6  -> 64-QAM

    Input:
        bits: 1D bit array
        mu: bits per symbol

    Output:
        complex-valued modulation symbols
    """
    bit_r = bits.reshape((int(len(bits) / mu), mu))

    if mu == 2:
        # QPSK
        real = 2 * bit_r[:, 0] - 1
        imag = 2 * bit_r[:, 1] - 1
        return (real + 1j * imag) / np.sqrt(2)

    elif mu == 6:
        # 64-QAM
        # 3 bits for I branch, 3 bits for Q branch
        # Mapping: 000~111 -> -7, -5, -3, -1, +1, +3, +5, +7
        i_bits = bit_r[:, 0:3].astype(int)
        q_bits = bit_r[:, 3:6].astype(int)

        i_index = i_bits[:, 0] * 4 + i_bits[:, 1] * 2 + i_bits[:, 2]
        q_index = q_bits[:, 0] * 4 + q_bits[:, 1] * 2 + q_bits[:, 2]

        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])

        real = levels[i_index]
        imag = levels[q_index]

        # Average power normalization for square 64-QAM
        return (real + 1j * imag) / np.sqrt(42)

    else:
        raise ValueError("Only mu=2 QPSK and mu=6 64-QAM are supported.")


# Replace the original Modulation function in utils.py
utils.Modulation = Modulation


# =========================================================
# Basic Config
# =========================================================
SEED = 42

K = 64
CP = 16
CHANNEL_LEN = 16

MU = 6                      # 64-QAM
N_BITS = K * MU             # 64 * 6 = 384 bits

# Task (c): 64-QAM
SNR_LIST = [5, 10, 15, 20, 25]

# You can change this depending on the experiment.
# If the assignment only asks 64-QAM, using one pilot setting is enough.
# To compare pilot effect under 64-QAM, keep multiple pilot settings.
PILOT_LIST = [0,8, 16, 64]

# According to Task (c):
# main.py: config.pred_range = np.arange(48, 96)
# Train.py: n_output = 48
PRED_RANGE = np.arange(48, 96)
N_OUTPUT = len(PRED_RANGE)  # 48

TRAIN_SAMPLES = 20000
TEST_SAMPLES = 100000
EPOCHS = 20000
BATCH_SIZE = 512
LR = 1e-3

CP_FLAG = True
CLIPPING_FLAG = False

SAVE_DIR = "./task_c_64qam_pytorch_results"
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
    Generate pilotCarriers, dataCarriers, and pilotValue
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
        # Therefore pilotValue should be a length-K vector.
        pilot_value = np.ones(K, dtype=complex) * (3 + 3j)

    else:
        pilot_carriers = np.linspace(0, K - 1, P, dtype=np.int64)
        pilot_carriers = np.unique(pilot_carriers)
        data_carriers = np.setdiff1d(all_carriers, pilot_carriers)

        # In utils.py, if P < K:
        #     OFDM_data[pilotCarriers] = pilotValue
        # A scalar pilot value is valid.
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
        feature from utils.ofdm_simulate()
        shape = [num_samples, 256] when K = 64

    y:
        selected target bits according to PRED_RANGE
        shape = [num_samples, 48] for Task (c)
    """
    pilot_carriers, data_carriers, pilot_value = make_pilot_config(P)

    X_list = []
    y_list = []

    for _ in range(num_samples):
        # For 64-QAM, codeword length = K * MU = 384 bits
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
def normalize_train_test(X_train, X_test):
    """
    Standardize input features using training-set mean and std.
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
# Main Task (c)
# =========================================================
def main():
    ber_results = {}

    for P in PILOT_LIST:
        ber_results[P] = []

        for snr_db in SNR_LIST:
            print("\n" + "=" * 80)
            print(f"Task (c) 64-QAM | SNR = {snr_db} dB | Pilots = {P}")
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
                f"FCDNN_64QAM_SNR{snr_db}_Pilot{P}_out{N_OUTPUT}.pth"
            )

            print("Training model...")
            train_model(X_train, y_train, model_path)

            print("Testing model...")
            ber = test_model(X_test, y_test, model_path)

            ber_results[P].append(ber)

            print(
                f"Result | 64-QAM | SNR = {snr_db} dB | "
                f"Pilots = {P} | BER = {ber:.8f}"
            )

    # Save txt results
    txt_path = os.path.join(SAVE_DIR, "task_c_64qam_ber_results.txt")

    with open(txt_path, "w") as f:
        f.write("Task (c) 64-QAM BER Results\n")
        f.write(f"MU = {MU}\n")
        f.write(f"N_BITS = {N_BITS}\n")
        f.write(f"PRED_RANGE = {PRED_RANGE[0]} to {PRED_RANGE[-1]}\n")
        f.write(f"N_OUTPUT = {N_OUTPUT}\n\n")

        for P in PILOT_LIST:
            label = "No Pilot" if P == 0 else f"Pilot = {P}"
            f.write(label + "\n")

            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"SNR = {snr_db} dB, BER = {ber:.8f}\n")

            f.write("\n")

    print("\nSaved BER results to:", txt_path)

    # Save CSV results
    csv_path = os.path.join(SAVE_DIR, "task_c_64qam_ber_results.csv")

    with open(csv_path, "w") as f:
        f.write("Pilot,SNR,BER\n")
        for P in PILOT_LIST:
            for snr_db, ber in zip(SNR_LIST, ber_results[P]):
                f.write(f"{P},{snr_db},{ber:.10f}\n")

    print("Saved BER CSV to:", csv_path)

    # Plot BER vs SNR
    plt.figure(figsize=(7, 5))

    for P in PILOT_LIST:
        label = "No Pilot" if P == 0 else f"Pilot = {P}"

        # Avoid log(0)
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
    plt.title("Task (c): 64-QAM BER vs. SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(SAVE_DIR, "task_c_64qam_BER_vs_SNR.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print("Saved BER figure to:", fig_path)


if __name__ == "__main__":
    main()