import functions as fn
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torchvision import transforms
import cv2


print('Estandarizando train...')
train = pd.read_csv('./train.csv')
train_list_rgb = fn.get_images_list(train,'rgb')
norm_params_rgb_mean, norm_params_rgb_std = fn.load_image_normalize_rgb(train_list_rgb, "./data_img/", (36,108))

with open('valores_mean_std.txt', 'w') as f:
    f.write(",".join(map(str, norm_params_rgb_mean)) + "\n")
    f.write(",".join(map(str, norm_params_rgb_std)) + "\n")
