import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import h5py
import random
import math
from tqdm import tqdm

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

AES_Sbox_inv =  np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def HW(s):
    return bin(s).count("1")


def aes_label_cpa_ascad(plt, correct_key, leakage, mask):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key] ^ mask[i])
        elif leakage == 'ID':
            # print(f"mask:{mask} plt:{plt} correct_key:{correct_key}")
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key] ^ mask[i]
            # exit()
    return labels_for_snr


def aes_label_cpa(plt, correct_key, leakage):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
    return labels_for_snr

def aes_label_cpa_mask(dataset, plt, correct_key, leakage, mask, order = 1):

    num_traces = plt.shape[0]

    labels_for_snr = np.zeros(num_traces)
    if order == 0:
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
    elif order == 1:
        if dataset == "AES_HD_ext_plaintext":
            pt_15 = mask[:, 0]
            pt_11 = mask[:, 1]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i]
        elif dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label":
            pt_15 = plt
            pt_11 = mask[:, 1]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i]
        else:
            if "simulated" in dataset:
                # mask1 = mask[0] #for original
                mask1 = mask[:, 0]
            else:
                mask1 = mask[:, 0]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]
    elif order == 2:
        if "simulated" in dataset:
            # mask1 = mask[0]  #for original
            # mask2 = mask[1] #for original
            mask1 = mask[: ,0]
            mask2 = mask[: ,1]
        else:
            mask1 = mask[: ,0]
            mask2 = mask[: ,1]
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]
    elif order == 3:
        if "simulated" in dataset:
            # mask1 = mask[0] #for original
            # mask2 = mask[1] #for original
            # mask3 = mask[2] #for original
            mask1 = mask[:, 0]
            mask2 = mask[: ,1]
            mask3 = mask[:, 2]
        else:
            mask1 = mask[:, 0]
            mask2 = mask[: ,1]
            mask3 = mask[:, 2]
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]^mask3[i])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]^mask3[i]
    return labels_for_snr


def cpa_method(total_samplept, number_of_traces, label, traces):
    if total_samplept == 1:
        cpa = np.zeros(total_samplept)
        cpa[0] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces])[1, 0])
    else:
        cpa = np.zeros(total_samplept)
        for t in range(total_samplept):
            # print(f'corrcof:{np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])}')
            cpa[t] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])[1, 0])
    return cpa

def load_ascad(ascad_database_file, leakage_model='HW', byte = 2,train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):
    in_file = h5py.File(ascad_database_file, "r")

    # 读取曲线
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    # print(X_profiling.shape)
    # X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # print(X_profiling.shape)
    # exit()
    # 明文
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])
    # mask
    P_masks_15 = np.array(in_file['Profiling_traces/metadata'][:]['masks'][:, 15])
    P_masks = np.array(in_file['Profiling_traces/metadata'][:]['masks'][:, byte])

    # 如果 byte 不等于 2 ，那么就对明文进行 Sbox 变换
    if byte != 2:
        key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:,byte])
        Y_profiling = np.zeros(P_profiling.shape[0])
        print("Loading Y_profiling")
        for i in tqdm(range(len(P_profiling))):
            Y_profiling[i] = AES_Sbox[P_profiling[i] ^ key_profiling[i]]
        if leakage_model == 'HW':
            Y_profiling = calculate_HW(Y_profiling)
    else:
        Y_profiling = np.array(in_file['Profiling_traces/labels'])  # This is only for byte 2
        if leakage_model == 'HW':
            Y_profiling = calculate_HW(Y_profiling)

    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'])
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))

    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte])
    attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    profiling_key = np.array(in_file['Profiling_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    print(f'attack_key:{attack_key}')
    print(f'profile_key:{profiling_key}')

    if byte != 2:
        print("Loading Y_attack")
        key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:,byte])
        Y_attack = np.zeros(P_attack.shape[0])
        for i in tqdm(range(len(P_attack))):
            Y_attack[i] = AES_Sbox[P_attack[i] ^ key_attack[i]]
        if leakage_model == 'HW':
            Y_attack = calculate_HW(Y_attack)
    else:
        Y_attack = np.array(in_file['Attack_traces/labels'])
        if leakage_model == 'HW':
            Y_attack = calculate_HW(Y_attack)

    print("Information about the dataset: ")
    print("X_profiling total shape", X_profiling.shape)
    if leakage_model == 'HW':
        print("Y_profiling total shape", len(Y_profiling))
    else:
        print("Y_profiling total shape", Y_profiling.shape)
    print("Plt_profiling total shape", P_profiling.shape)
    print("X_attack total shape", X_attack.shape)
    if leakage_model == 'HW':
        print("Y_attack total shape", len(Y_attack))
    else:
        print("Y_attack total shape", Y_attack.shape)
    print("Plt_attack total shape", P_attack.shape)
    print("correct key:", attack_key)
    print()

    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key, P_masks, P_masks_15


def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]