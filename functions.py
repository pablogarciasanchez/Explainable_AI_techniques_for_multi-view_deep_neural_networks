import enum
from random import random
from re import S
from xml.etree.ElementInclude import include
from sklearn import metrics
import torch
from torchvision.io import read_image
from torchvision import transforms
from sklearn.model_selection import KFold
from collections import Counter
import scipy.ndimage as sci
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2 as cv

import random

import ranges_of_age as roa

import sys

from datetime import datetime

def load_image_normalize_rgb(images, dir, img_shape):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_shape),
        transforms.ToTensor()
    ])

    r_channel_list = []
    g_channel_list = []
    b_channel_list = []
    
    for img in images:
        for i in img[:3]:
            image = transform(read_image(dir + i))

            # Separar los canales y agregar a la lista correspondiente
            r_channel_list.append(image[0])
            g_channel_list.append(image[1])
            b_channel_list.append(image[2])

    r_channel_tensor = torch.stack(r_channel_list)
    g_channel_tensor = torch.stack(g_channel_list)
    b_channel_tensor = torch.stack(b_channel_list)

    # Calcular la media y la desviación estándar para cada canal
    r_mean = torch.mean(r_channel_tensor).item()
    g_mean = torch.mean(g_channel_tensor).item()
    b_mean = torch.mean(b_channel_tensor).item()
    
    r_std = torch.std(r_channel_tensor).item()
    g_std = torch.std(g_channel_tensor).item()
    b_std = torch.std(b_channel_tensor).item()

    #print(r_mean, r_std)
    #print(g_mean, g_std)
    #print(b_mean, b_std)

    normalize = transforms.Normalize([r_mean, g_mean, b_mean], [r_std, g_std, b_std])

    normalized_images = []
    for img in images:
        for i in img[:3]:
            image = transform(read_image(dir + i))
            image = normalize(image)
            
            normalized_images.append(image)

    normalized_images_tensor = torch.stack(normalized_images)

    #print('Media final', torch.mean(normalized_images_tensor).item(), torch.std(normalized_images_tensor).item())

    return [r_mean, r_std, g_mean], [g_std, b_mean, b_std]


def get_images_list(dataframe, colormode, augment=True, weights=True):
    data_list = dataframe.values

    if colormode == 'rgb':
        img_names = ['_panorama_ext_X.png', '_panorama_ext_Y.png', '_panorama_ext_Z.png']
    elif colormode == 'grayscale':
        img_names = ['_panorama_SDM.png', '_panorama_NDM.png', '_panorama_GNDM.png']

    if augment:
        image_list = [
            [f"{t[1]}/{t[1]}_{n}{name}" for name in img_names] + [t[2], t[3]]
            for t in data_list for n in range(20)
        ]
    else:
        image_list = [
            [f"{t[1]}/{t[1]}_0{name}" for name in img_names] + [t[2], t[3]]
            for t in data_list
        ]

    return np.array(image_list)

def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

def bin(age, age_list):
    unique_list = []
    for i in np.arange(np.max(age_list)+1):
        if i not in unique_list:
            unique_list.append(i)

    return unique_list.index(age)

def inverse(x):
    if x != 0:
        v = np.float32(1/x)
    else:
        v = np.float32(1)
    return v

def calculate_weights(sample_df):
    age_list = sample_df['Edad'].to_list()
    bin_index_per_label = [bin(label,age_list) for label in age_list]
    
    N_ranges = max(bin_index_per_label) + 1
    num_samples_of_bin = dict(Counter(bin_index_per_label))

    emp_label_dist = [num_samples_of_bin.get(i,0) for i in np.arange(N_ranges)]
   
    lds_kernel = cv.getGaussianKernel(5,2).flatten()
    eff_label_dist = sci.convolve1d(np.array(emp_label_dist), weights=lds_kernel, mode='constant')
    
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    
    weights = [inverse(x) for x in eff_num_per_label]

    sample_df['Weight'] = weights

def range_of_age(n,ranges):
    for i,r in enumerate(ranges):
        if n >= r[0] and n <= r[1]:
            return i
        
def train_test_split(sample_df, size):
    dic_aux = {}
    sub_df = sample_df
    column = 'Number'
    column_number = 0

    range = np.unique(np.array(sub_df[column].to_list()))

    div = int(len(range) * size)
    if div == 0:
        div = 1

    np.random.shuffle(range)

    for i in range[:div]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Train'

    for i in range[div:]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Test'

    set_list = []

    for i in sample_df[column].values:
        set_list.append(dic_aux[i])
    
    sample_df.insert(0,'Set', set_list, True)

    train_df = sample_df[sample_df["Set"] == 'Train']
    train_df = train_df.drop(columns=["Set"])

    test_df = sample_df[sample_df["Set"] == 'Test']
    test_df = test_df.drop(columns=["Set"])

    return train_df, test_df

def precision_by_range(y_true, y_pred, ranges, metric='mae'):

    prec_dic = {}
    for r in np.arange(len(ranges)):
        prec_dic[r] = [[],[],[]]

    for yt, yp in zip(y_true, y_pred):
        r_age = range_of_age(yt, ranges)

        if metric == 'mae':
            dif = abs(yt - yp)
            metric_name = 'MAE'
        elif metric == 'mse':
            dif = (yt - yp) * (yt - yp)
            metric_name = 'MSE'
        prec_dic[r_age][0].append(dif)
        prec_dic[r_age][1].append(yp)
        prec_dic[r_age][2].append(yt)
    
    df_precision = pd.DataFrame()

    prec_dic_filter = {}
    for k in prec_dic.keys():
        if len(np.array(prec_dic[k][1])) != 0:
            prec_dic_filter[k] = prec_dic[k]

    df_precision['Range'] = [ranges[k] for k in prec_dic_filter.keys()]
    df_precision['N values'] = [len(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['T values Mean'] = [np.mean(np.array(prec_dic_filter[k][2])) for k in prec_dic_filter.keys()]
    df_precision['T values std'] = [np.std(np.array(prec_dic_filter[k][2])) for k in prec_dic_filter.keys()]
    df_precision['Values Mean'] = [np.mean(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['Values std'] = [np.std(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['Values 1%'] = [np.percentile(np.array(prec_dic_filter[k][1]),1) for k in prec_dic_filter.keys()]
    df_precision['Values 10%'] = [np.percentile(np.array(prec_dic_filter[k][1]),10) for k in prec_dic_filter.keys()]
    df_precision['Values 25%'] = [np.percentile(np.array(prec_dic_filter[k][1]),25) for k in prec_dic_filter.keys()]
    df_precision['Values 50%'] = [np.percentile(np.array(prec_dic_filter[k][1]),50) for k in prec_dic_filter.keys()]
    df_precision['Values 75%'] = [np.percentile(np.array(prec_dic_filter[k][1]),75) for k in prec_dic_filter.keys()]
    df_precision['Values 99%'] = [np.percentile(np.array(prec_dic_filter[k][1]),99) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' Mean'] = [np.mean(np.array(prec_dic_filter[k][0])) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' Std'] = [np.std(np.array(prec_dic_filter[k][0])) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 1%'] = [np.percentile(np.array(prec_dic_filter[k][0]),1) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 10%'] = [np.percentile(np.array(prec_dic_filter[k][0]),10) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 25%'] = [np.percentile(np.array(prec_dic_filter[k][0]),25) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 50%'] = [np.percentile(np.array(prec_dic_filter[k][0]),50) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 75%'] = [np.percentile(np.array(prec_dic_filter[k][0]),75) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' 99%'] = [np.percentile(np.array(prec_dic_filter[k][0]),99) for k in prec_dic_filter.keys()]

    return df_precision

def show_stats(true,pred,metric_range='mae'):
    stats_mae = abs(true-pred)
    stats_mse = (true-pred)*(true-pred)
    measures = [stats_mae, stats_mse]

    df_stats = pd.DataFrame()
    df_stats['Metric'] = ['MAE' ,'MSE']
    df_stats['Mean:'] = [np.mean(m) for m in measures]
    df_stats['Std:'] = [np.std(m) for m in measures]
    df_stats['1% value:'] = [np.percentile(m,1) for m in measures]
    df_stats['10% value:'] = [np.percentile(m,10) for m in measures]
    df_stats['25% value:'] = [np.percentile(m,25) for m in measures]
    df_stats['50% value:'] = [np.median(m) for m in measures]
    df_stats['75% value:'] = [np.percentile(m,75) for m in measures]
    df_stats['99% value:'] = [np.percentile(m,99) for m in measures]
    df_stats['Min value:'] = [np.min(m) for m in measures]
    df_stats['Max value:'] = [np.max(m) for m in measures]

    ranges = [roa.ranges_todd,roa.ranges_5,roa.ranges_3]
    for range in ranges:
        df_precision = precision_by_range(true, pred, range, metric_range)
        if metric_range == 'mae':
            print('Mean of means:', np.mean(df_precision['MAE Mean'].to_numpy()))
        elif metric_range == 'mse':
            print('Mean of means:', np.mean(df_precision['MSE Mean'].to_numpy()))
        print(df_precision.to_string())

    print(df_stats.to_string())
