import os
import numpy as np
import matplotlib.pyplot as plt

from Train import train
from Test import test


class sysconfig(object):
    Pilots = 8
    with_CP_flag = True
    SNR = 20
    Clipping = False

    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models/'

    # QPSK original setting
    pred_range = np.arange(16, 32)

    learning_rate = 0.001
    learning_rate_decrease_step = 2000

    # Do not manually assign model.ckpt
    model_name = None


def plot_ber_results(SNR_list, Pilot_list, all_results, save_path):
    plt.figure(figsize=(8, 6))

    for pilot_num in Pilot_list:
        label = "No Pilot" if pilot_num == 0 else "Pilots = {}".format(pilot_num)

        plt.semilogy(
            SNR_list,
            all_results[pilot_num],
            marker='o',
            linewidth=2,
            label=label
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs. SNR with Different Number of Pilots")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    # ==============================
    # Task (b) setting
    # ==============================
    SNR_list = np.arange(5, 26, 5)

    # 
    Pilot_list = [8, 16, 64]

    # True  = train + test
    # False = only test existing checkpoints
    IS_Training = True

    all_results = {}

    for pilot_num in Pilot_list:
        all_results[pilot_num] = []

        for snr in SNR_list:
            config = sysconfig()

            config.Pilots = pilot_num
            config.SNR = snr

            # 
            config.Model_path = '../Models/Pilot_{}_SNR_{}/'.format(pilot_num, snr)

            if not os.path.exists(config.Model_path):
                os.makedirs(config.Model_path)

            # 
            config.model_name = None

            print("\n" + "=" * 70)
            print("Task (b) Running")
            print("Pilots =", pilot_num)
            print("SNR =", snr, "dB")
            print("Model path =", config.Model_path)
            print("=" * 70)

            if IS_Training:
                train(config)

            ber = test(config)

            print("Final Result | Pilots = {} | SNR = {} dB | BER = {}".format(
                pilot_num, snr, ber
            ))

            all_results[pilot_num].append(ber)

    print("\n========== Task (b) BER Results ==========")

    for pilot_num in Pilot_list:
        label = "No Pilot" if pilot_num == 0 else "Pilots = {}".format(pilot_num)
        print(label, ":", all_results[pilot_num])

    plot_ber_results(
        SNR_list=SNR_list,
        Pilot_list=Pilot_list,
        all_results=all_results,
        save_path="task_b_BER_vs_SNR.png"
    )


if __name__ == '__main__':
    main()