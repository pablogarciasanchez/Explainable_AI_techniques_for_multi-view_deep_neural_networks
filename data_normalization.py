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
norm_params_rgb = fn.load_image_normalize_rgb(train_list_rgb, "./data_img/", (36,108))
print(norm_params_rgb)

# Definir las medias y desviaciones estándar de los datos de entrenamiento
mean = [0.09538473933935165, 0.06967461854219437, 0.04578746110200882]
std = [0.20621216297149658, 0.15044525265693665, 0.088300921022892]