import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw, ImageFont
from captum.attr import LayerGradCam, LayerAttribution, visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import functions as fn
import cv2
from torchinfo import summary
import sys
import torchvision.models as models
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import Saliency
sys.stdout.reconfigure(encoding='utf-8')

from utils import PanoramaCNN, ResnetCNN, PanoramaCNNEx, ResnetCNNEx, CustomImageDataset

def custom_color_map(value):
    if value >= 0:
        return (255, 255 - int(255 * value), 255 - int(255 * value))  # Interpolación entre blanco y rojo
    else:
        return (255 + int(255 * value), 255 + int(255 * value), 255)  # Interpolación entre blanco y azul

def apply_color_map_to_data(data):
    colored_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            colored_data[i, j] = custom_color_map(data[i, j])
    return colored_data

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.5):

    # Normalizar el heatmap para que los valores estén entre -1 y 1
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 2 - 1
    
    # Cambiar el tamaño del mapa de calor para que coincida con el de la imagen original
    heatmap_resized = cv2.resize(heatmap_normalized, (original_image.width, original_image.height))
    
    # Aplicar el mapeo de colores personalizado
    heatmap_colored = apply_color_map_to_data(heatmap_resized)
    
    # Convertir la imagen PIL a array numpy y convertir a 3 canales (color)
    original_image_array = np.array(original_image)
    original_image_colored = cv2.cvtColor(original_image_array, cv2.COLOR_GRAY2RGB)
    
    # Superponer el mapa de calor en la imagen original
    superimposed_image = cv2.addWeighted(original_image_colored, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convertir el array numpy de la imagen superpuesta a formato PIL
    superimposed_image_pil = Image.fromarray(superimposed_image)

    return superimposed_image_pil

def overlay_heatmap_on_image_gradcam(heatmap, original_image, alpha=0.5):

    # Cambiar el tamaño del mapa de calor para que coincida con el de la imagen original
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))

    # Aplicar el mapa de calor JET y convertirlo a RGB
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Convertir la imagen PIL a un array de numpy y asegurarse de que sea RGB
    original_rgb = np.array(original_image)
    if len(original_rgb.shape) == 2:  # Si es escala de grises
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)

    # Fusionar el mapa de calor y la imagen original
    fusionada = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored_rgb, alpha, 0)

    # Convertir el arreglo de NumPy de nuevo a imagen PIL y retornarlo
    return Image.fromarray(fusionada)

# Funciones auxiliares
def load_image(image_path, preprocess):
    """Carga y preprocesa una imagen."""
    image = Image.open(image_path)
    # Dividir la imagen en canales
    r, g, b = image.split()

    return g,preprocess(image)

def get_gradcam(images, target_layer, model):
    """Genera Grad-CAM atribuciones para imágenes."""
    
    model.zero_grad()
    grad_cam = LayerGradCam(model, target_layer)
    attributions = grad_cam.attribute(images, relu_attributions=True)

    upsampled_attr = LayerAttribution.interpolate(attributions, (36, 108))

    return upsampled_attr

def get_attribution_channel(attribution, channel):
    return attribution[channel].detach().cpu().numpy()

def test(model):
    print("Cargando conjunto de test...")
    test_data = CustomImageDataset("test.csv","data_img",mean_train, std_train, False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    model.eval()
    true = []
    pred = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            features, labels = data[0].to(device), data[1].to(device)
            outputs = model(features).view(-1)
            true.extend(labels.cpu().numpy().flatten().tolist())
            pred.extend(outputs.cpu().numpy().flatten().tolist())
    true = np.array(true)
    pred = np.array(pred)

    fn.show_stats(true,pred,'mae')
    fn.show_stats(true,pred,'mse')

def save_heatmap(output_dir, subfolder, name, data):
    heatmap_path = os.path.join(output_dir, subfolder,name)
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    cv2.imwrite(heatmap_path, data)

def normalized(channel_data):
    # Normalizar el canal de datos usando los valores mínimos y máximos encontrados
    norm_data = 255 * (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
    norm_data = norm_data.astype(np.uint8)
    return norm_data

def apply_smoothgrad(images, model, subfolder, image_wo_p):
    ig = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(ig)

    attribution, delta = noise_tunnel.attribute(images, nt_type='smoothgrad', nt_samples=50, nt_samples_batch_size=1, stdevs=0.1,
                                                     return_convergence_delta=True, n_steps=150)
    
    attribution = attribution.squeeze()

    channels = ['X', 'Y', 'Z']
    channel_to_index = {'X': 0, 'Y': 1, 'Z': 2}  # Mapeo de canales a índices

    attributions = {channel: get_attribution_channel(attribution, idx)[1] for idx, channel in enumerate(channels)}

    for channel_name, channel_data in attributions.items():
        i = channel_to_index[channel_name]  # Obtiene el valor de i según channel_name

        normalized_data = normalized(channel_data)
        heatmap_img = overlay_heatmap_on_image(normalized_data, images_wo_p[i], 0.6)
        save_heatmap(output_dir, subfolder, f'Smoothgrad_cnn{channel_name}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'Smoothgrad_{channel_name}_heatmap.png'))

def apply_integratedgradients(images, model, subfolder, image_wo_p):
    ig = IntegratedGradients(model)
    attribution,delta = ig.attribute(images,return_convergence_delta=True,n_steps=200)
        
    attribution = attribution.squeeze()

    channels = ['X', 'Y', 'Z']
    channel_to_index = {'X': 0, 'Y': 1, 'Z': 2}  # Mapeo de canales a índices

    attributions = {channel: get_attribution_channel(attribution, idx)[1] for idx, channel in enumerate(channels)}

    for channel_name, channel_data in attributions.items():
        i = channel_to_index[channel_name]  # Obtiene el valor de i según channel_name

        normalized_data = normalized(channel_data)
        heatmap_img = overlay_heatmap_on_image(normalized_data, images_wo_p[i], 0.6)
        save_heatmap(output_dir, subfolder, f'IntegratedGradients_cnn{channel_name}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'IntegratedGradients_{channel_name}_heatmap.png'))

def apply_saliency(images, model, subfolder, image_wo_p):
    saliency = Saliency(model)
    attribution = saliency.attribute(images, abs=False)
        
    attribution = attribution.squeeze()

    channels = ['X', 'Y', 'Z']
    channel_to_index = {'X': 0, 'Y': 1, 'Z': 2}  # Mapeo de canales a índices

    attributions = {channel: get_attribution_channel(attribution, idx)[1] for idx, channel in enumerate(channels)}

    for channel_name, channel_data in attributions.items():
        i = channel_to_index[channel_name]  # Obtiene el valor de i según channel_name

        normalized_data = normalized(channel_data)
        heatmap_img = overlay_heatmap_on_image(normalized_data, image_wo_p[i], 0.6)
        save_heatmap(output_dir, subfolder, f'Saliency_cnn{channel_name}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder, f'Saliency_{channel_name}_heatmap.png'))


def apply_gradcam(images, model, is_panorama, subfolder, image_wo_p):
    cnn_names = ["cnnX", "cnnY", "cnnZ"]
    for i, cnn in enumerate(cnn_names):
        if is_panorama:
            target_layer = model.__getattr__(cnn)[-10]
        else:
            target_layer = model.__getattr__(cnn)[0][-1][-1].conv3

        attributions = get_gradcam(images, target_layer, model)
        
        attributions = attributions.squeeze(0, 1).detach().cpu().numpy()
        
        # Normalizar el canal de datos usando los valores mínimos y máximos encontrados
        normalized_data = normalized(attributions)
        heatmap_img = overlay_heatmap_on_image_gradcam(normalized_data, images_wo_p[i], 0.6)
        save_heatmap(output_dir, subfolder, f'GradCam_{cnn}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'GradCam_{cnn}_heatmap.png'))
    
# PARÁMETROS
PANORAMACNN = True
Explainable = True

mean_train = [0.09538473933935165, 0.06967461854219437, 0.04578746110200882]
std_train = [0.20621216297149658, 0.15044525265693665, 0.088300921022892]
image_size = (36,108)
image_dir = './data_img'
image_files = ['_0_panorama_ext_X.png', '_0_panorama_ext_Y.png', '_0_panorama_ext_Z.png']

# Configuración del modelo
if(Explainable):
    path = './entrenamientos/PanoramaCNNEx/' if PANORAMACNN else './entrenamientos/ResnetCNNEx/'
else:
    path = './entrenamientos/PanoramaCNN/' if PANORAMACNN else './entrenamientos/ResnetCNN/'
path += 'entrenamiento_231003_1746'


output_dir = './heatmap' + ('PanoramaCNN/' if PANORAMACNN else 'ResnetCNN/')
if(Explainable):
    output_dir = './heatmap' + ('PanoramaCNNEx/' if PANORAMACNN else 'ResnetCNNEx/')
else:
    output_dir = './heatmap' + ('PanoramaCNN/' if PANORAMACNN else 'ResnetCNN/')

if(Explainable):
    model = PanoramaCNNEx() if PANORAMACNN else ResnetCNNEx()
else:
    model = PanoramaCNN() if PANORAMACNN else ResnetCNN()


print(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.load_state_dict(torch.load(path))  # Load the parameters from the saved state_dict
model.to(device)  # Move the model to the desired device

preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((36,108),antialias=True),
            transforms.Normalize(mean=mean_train, std=std_train)
        ])

model.eval()
#test(model)


df = pd.read_csv('./test.csv', delimiter=',')
image_dir = df['Sample'].tolist()

new_df = df[['Sample', 'Edad']].copy()

predicted_ages_dict = {}
saliency_values_dict = {}
for subfolder in image_dir:

    print(f"Procesando {subfolder}...")

    images_wo_p, images = zip(*[load_image(os.path.join('./data_img', subfolder, f'{subfolder}{file}'), preprocess) for file in image_files if os.path.exists(os.path.join('./data_img', subfolder, f'{subfolder}{file}'))])

    if images:
        images = torch.stack(images).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_age = model(images).item()
            x_saliency_value = model.view_saliency_layer.output

        predicted_ages_dict[subfolder] = predicted_age
        x_saliency_list = x_saliency_value.tolist()
        saliency_values_dict[subfolder] = x_saliency_list[0]

        images.requires_grad = True

        #apply_saliency(images,model, subfolder, images_wo_p)

        #apply_smoothgrad(images, model, subfolder, images_wo_p)

        #apply_integratedgradients(images,model, subfolder, images_wo_p)

        apply_gradcam(images,model, PANORAMACNN, subfolder, images_wo_p)

'''
new_df['Edad_predicha'] = new_df['Sample'].map(predicted_ages_dict)
new_df['Diferencia'] = (new_df['Edad'] - new_df['Edad_predicha']).abs()
new_df['Indicador'] = (new_df['Diferencia'] < 1.5).astype(int)

new_df['ponderacionX'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[0])
new_df['ponderacionY'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[1])
new_df['ponderacionZ'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[2])

new_df.to_csv('./predicciones.csv', index=False)
'''