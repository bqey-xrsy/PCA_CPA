import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ascad

import torch
from sklearn.decomposition import PCA

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

class Custom_Dataset_Profiling(Dataset):
    def __init__(self,nb_traces_attacks, root = './', dataset = "ASCAD", leakage = "ID",transform = None, embedding_size = None):
        self.nb_traces_attacks = nb_traces_attacks
        if dataset == "ASCAD":
            byte = 2
            data_root = 'Dataset/ASCAD.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key, self.P_masks, self.P_masks_15 = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
                test_end=self.nb_traces_attacks)
            self.X_profiling_orig = self.X_profiling
            self.X_attack_orig = self.X_attack
            self.X_profiling = np.pad(self.X_profiling, ((0, 0), (2, 2)), 'constant', constant_values=(0, 0))
            self.X_attack = np.pad(self.X_attack, ((0, 0), (2, 2)), 'constant', constant_values=(0, 0))
            self.order = 1
     
        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.scaler = MinMaxScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)
        self.dataset = dataset

    def apply_MinMaxScaler(self):
        self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        self.X_attack = self.scaler.transform(self.X_attack)
        print("After minmaxscaler X_profiling max:", np.max(self.X_profiling))
        print("After minmaxscaler X_profiling min:", np.min(self.X_profiling))

    def PCA(self, n_components):
        self.PCA = PCA(n_components=n_components)
        self.PCA.fit(self.X_profiling)
        self.X_profiling =self.PCA.transform(self.X_profiling)
        self.X_attack = self.PCA.transform(self.X_attack)
        print("PCA X_profiling ", self.X_profiling.shape)
        print("PCA X_attack ", self.X_attack.shape)

    def split_profiling_train_val(self, X, Y):
        self.X_profiling_train, self.X_profiling_val, self.Y_profiling_train, self.Y_profiling_val = train_test_split(
            X, Y, test_size=0.1, random_state=0)

    def choose_phase(self,phase):
        if phase == 'train_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling, 1), self.Y_profiling, self.Y_profiling

        elif phase == 'test_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_attack, 1), self.Y_attack, self.Y_attack

        elif phase == 'train_masking':

            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling, 1), self.Y_profiling, self.Y_profiling
            for i in range(len(self.X_profiling)):
                self.Y[i] = AES_Sbox[self.plt_profiling[i] ^ self.correct_key] ^ self.P_masks_15[i]
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive, 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, label, plaintext= sample['trace'], sample['sensitive'], sample['plaintext']

        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long(), torch.from_numpy(np.array(plaintext)).long()

