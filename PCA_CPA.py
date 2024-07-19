import os
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from src.dataloader import ToTensor_trace, Custom_Dataset_Profiling
from src.utils import cpa_method, aes_label_cpa_ascad


dataset = "ASCAD" #simulated_traces_order_3 #Chipwhisperer #AES_HD_ext_plaintext #AES_HD_ext_sbox #AES_HD_ext_label #latent_simulated_traces_order_0
leakage = "ID"

nb_traces_attacks = 1000 
batch_size = 200 #ASCADf/r: 200

root = "./"

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_root = "./PCA_CPA_images/"
if not os.path.exists(image_root):
    os.makedirs(image_root)

embedding_size = None
dataloadertrain = Custom_Dataset_Profiling(nb_traces_attacks, root = root, dataset = dataset, leakage = leakage,embedding_size = embedding_size, transform =  transforms.Compose([ ToTensor_trace() ]))

# CPA
trace_size_original = dataloadertrain.X_profiling.shape[1]
used_traces_real = 10000


cpa_real_trace = True
# 计算并绘制真实曲线和密钥的 CPA 曲线
if cpa_real_trace == True:
    fig, ax = plt.subplots(figsize=(15, 7))
    x_axis = [i for i in range(trace_size_original)]

    real_traces = np.zeros((used_traces_real, trace_size_original)) #全 0 数组 用于存储真实traces
    for i, trace in enumerate(dataloadertrain.X_profiling[:used_traces_real,:]):
        real_traces[i,:] = trace
    # 如果当前密钥值等于 correct_key，则绘制红色曲线，否则绘制灰色曲线。
    cpa_k_mask = cpa_method(trace_size_original, used_traces_real, dataloadertrain.P_masks[:used_traces_real], real_traces)
    for k in range(256):
        label_k_16_mask = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces_real], k, leakage, dataloadertrain.P_masks_15[:used_traces_real])
        cpa_k_16_mask = cpa_method(trace_size_original, used_traces_real, label_k_16_mask, real_traces)
        
        cpa_k = abs(cpa_k_16_mask - cpa_k_mask)

        if dataloadertrain.correct_key == k:
            continue
        else:
            ax.plot(x_axis, cpa_k, c="grey")

    label_correct_key = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces_real], dataloadertrain.correct_key, leakage, dataloadertrain.P_masks_15[:used_traces_real])
    cpa_k = cpa_method(trace_size_original, used_traces_real,  label_correct_key,  real_traces)

    # ax.plot(x_axis, cpa_k_mask, c="blue")

    ax.set_xlabel('Time Samples points', fontsize=20)
    ax.set_ylabel('Correlation(absolute)', fontsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    ax.plot(x_axis, cpa_k, c="red")
    plt.savefig(image_root + 'Corr_Actual_' + dataset + "_" + leakage+ ".png", dpi = 300)
    plt.close()
    print("real traces cpa caculate done")

# PCA
for n_conponents in [24, 48, 56]: #range(8,100,8):##ASCAD: [24] ASCAD_variable: [16] ASCAD_desync50:[48] ASCAD_variable_desync50:[24] #range(8,50,8): #Must be divisible by 8
   
    embedding_size = None
    dataloadertrain = Custom_Dataset_Profiling(nb_traces_attacks, root = root, dataset = dataset, leakage = leakage,embedding_size = embedding_size, transform =  transforms.Compose([ ToTensor_trace() ]))

    used_traces = 10000 #用于训练的trace数

    dataloadertrain.PCA(n_conponents)
    
    dataloadertrain.choose_phase("train_label")


    total_samplept = dataloadertrain.X_profiling.shape[1] #总采样点数
    total_num_trace =  dataloadertrain.X_profiling.shape[0] #生成的轨迹数

    print("total_samplept: ", total_samplept)
    print("total_num_trace: ", total_num_trace)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    x_axis = [i for i in range(total_samplept)]
    for i, trace in enumerate(dataloadertrain.X_profiling[:100,:]):
        ax.plot(x_axis, trace)
    ax.set_xlabel('Sample points', fontsize=20)
    ax.set_ylabel('Voltage', fontsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + 'Orig_traces_' + dataset + "_" + leakage+ "_" +str(n_conponents) + "_PCA.png")


    fig, ax = plt.subplots(figsize=(15, 7))
    x_axis = [i for i in range(total_samplept)]

    traces = np.zeros((used_traces, dataloadertrain.X_profiling.shape[1])) #全 0 数组 用于存储真实traces
    for i, trace in enumerate(dataloadertrain.X_profiling[:used_traces,:]):
        traces[i,:] = trace
    # 如果当前密钥值等于 correct_key，则绘制红色曲线，否则绘制灰色曲线。
    cpa_k_mask = cpa_method(total_samplept, total_num_trace, dataloadertrain.P_masks[:used_traces], traces)
    for k in range(256):
        label_k_16_mask = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], k, leakage, dataloadertrain.P_masks_15[:used_traces])
        cpa_k_16_mask = cpa_method(total_samplept, total_num_trace, label_k_16_mask, traces)
        
        cpa_k = abs(cpa_k_16_mask - cpa_k_mask)
    
        if dataloadertrain.correct_key == k:
            continue
        else:
            ax.plot(x_axis, cpa_k, c="grey")

    label_correct_key = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], dataloadertrain.correct_key, leakage, dataloadertrain.P_masks_15[:used_traces])
    cpa_k = cpa_method(total_samplept, total_num_trace,  label_correct_key,  traces)
    

    ax.set_xlabel('Time Samples points', fontsize=20)
    ax.set_ylabel('Correlation(absolute)', fontsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    ax.plot(x_axis, cpa_k, c="red")
    plt.savefig(image_root + 'Corr_' + dataset + "_" + leakage + "_" + str(n_conponents) + "_PCA_CPA.png", dpi = 300)

