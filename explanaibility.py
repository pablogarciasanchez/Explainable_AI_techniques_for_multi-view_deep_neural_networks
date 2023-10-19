import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import functions as fn
import sys
from captum.attr import IntegratedGradients, LayerGradCam, NoiseTunnel

sys.stdout.reconfigure(encoding='utf-8')

from utils import PanoramaCNN, ResnetCNN, PanoramaCNNEx, ResnetCNNEx, CustomImageDataset

def overlay_heatmap_on_image(heatmap_red, heatmap_blue, original_image, alpha=0.5):
    '''Superpone el mapa JET de openCV sobre la imagen original del hueso'''

    # Cambiar el tamaño de los mapas de calor para que coincidan con el de la imagen original
    heatmap_red_resized = cv2.resize(heatmap_red, (original_image.width, original_image.height))
    heatmap_blue_resized = cv2.resize(heatmap_blue, (original_image.width, original_image.height))
    
    # Crear mapas de calor en rojo y azul
    heatmap_red_rgb = np.zeros((original_image.height, original_image.width, 3), dtype=np.uint8)
    heatmap_red_rgb[:, :, 0] = heatmap_red_resized  # Canal rojo

    heatmap_blue_rgb = np.zeros((original_image.height, original_image.width, 3), dtype=np.uint8)
    heatmap_blue_rgb[:, :, 2] = heatmap_blue_resized  # Canal azul

    # Fusionar los mapas de calor rojo y azul
    combined_heatmap = cv2.addWeighted(heatmap_red_rgb, 1, heatmap_blue_rgb, 1, 0)

    # Convertir la imagen PIL a un array de numpy y asegurarse de que sea RGB
    original_rgb = np.array(original_image)
    if len(original_rgb.shape) == 2:  # Si es escala de grises
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)

    # Fusionar el mapa de calor combinado y la imagen original
    fusionada = cv2.addWeighted(original_rgb, 1 - alpha, combined_heatmap, alpha, 0)

    # Convertir el arreglo de NumPy de nuevo a imagen PIL y devolverlo
    return Image.fromarray(fusionada).resize((224, 224))

def overlay_heatmap_on_image_gradfr(heatmap, original_image, alpha=0.5):
    '''Superpone el mapa azul/rojo sobre la imagen original del hueso'''

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

    # Convertir el arreglo de NumPy de nuevo a imagen PIL y devolverlo
    return Image.fromarray(fusionada).resize((224, 224))

def load_image(image_path, preprocess):
    """Carga y preprocesa una imagen."""
    image = Image.open(image_path)

    gray_image = image.convert('L')

    return gray_image,preprocess(image)

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
    # Redimensionar la imagen a 36x108
    resized_data = cv2.resize(data, (108, 36))

    heatmap_path = os.path.join(output_dir, subfolder, name)
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    cv2.imwrite(heatmap_path, resized_data)

def normalized(channel_data):
    # Normalizar el canal de datos usando los valores mínimos y máximos encontrados
    if channel_data.max() != channel_data.min():
        norm_data = 255 * (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        norm_data = norm_data.astype(np.uint8)
    else:
        norm_data = np.zeros_like(channel_data, dtype=np.uint8)
    return norm_data

def normalized_2(channel_data1, channel_data2):

    maximo = np.max(np.concatenate((channel_data1, channel_data2)))
    minimo = np.min(np.concatenate((channel_data1, channel_data2)))

    # Verificar si maximo y minimo son iguales
    if maximo == minimo:
        return np.zeros_like(channel_data1, dtype=np.uint8), np.zeros_like(channel_data2, dtype=np.uint8)

    # Normalizar el canal de datos usando los valores mínimos y máximos encontrados
    norm_data_pos = 255 * ((channel_data1 - minimo) / (maximo - minimo))
    norm_data_pos = norm_data_pos.astype(np.uint8)

    # Normalizar el canal de datos usando los valores mínimos y máximos encontrados
    norm_data_neg = 255 * ((channel_data2 - minimo) / (maximo - minimo))
    norm_data_neg = norm_data_neg.astype(np.uint8)

    return norm_data_pos, norm_data_neg

def separate_attributions(attr):
    positive_attr = np.maximum(0, attr)
    negative_attr = np.abs(np.minimum(0, attr))
    return positive_attr, negative_attr

def apply_smoothgrad(images, model, subfolder, image_wo_p):
    '''
    Aplica Smoothgrad al modelo

    Parámetros
    ----------
    images : tensor
        Tensor que contiene las imágenes de entrada (Vistas X,Y,Z)
    model : torch.nn.Module
        Modelo MVCNN preentrenado.
    subfolder : str
        Nombre del subdirectorio donde se guardarán las imágenes resultantes.
    image_wo_p : list
        Lista de imágenes originales del hueso para la superposición del mapa de calor.
    '''
    ig = IntegratedGradients(model)

    noise_tunnel = NoiseTunnel(ig)

    attribution, delta = noise_tunnel.attribute(images, nt_type='smoothgrad', nt_samples=40, nt_samples_batch_size=1, stdevs=0.15,
                                                     return_convergence_delta=True, n_steps=200)
    attribution = attribution.squeeze()

    channels = ['X', 'Y', 'Z']
    channel_to_index = {'X': 0, 'Y': 1, 'Z': 2}

    attributions = {channel: get_attribution_channel(attribution, idx)[1] for idx, channel in enumerate(channels)}

    attributions_pos = {}  # Diccionario para almacenar las atribuciones positivas
    attributions_neg = {}  # Diccionario para almacenar las atribuciones negativas
    
    for idx, channel in enumerate(channels):
        attr_channel = get_attribution_channel(attribution, idx)[1]
        pos, neg = separate_attributions(attr_channel)
        attributions_pos[channel] = pos
        attributions_neg[channel] = neg

    for (channel_name_pos, channel_data_pos), (channel_name_neg, channel_data_neg)  in zip(attributions_pos.items(),attributions_neg.items()):

        i = channel_to_index[channel_name_pos]

        normalized_data_pos, normalized_data_neg = normalized_2(channel_data_pos, channel_data_neg)
        heatmap_img = overlay_heatmap_on_image(normalized_data_pos, normalized_data_neg, images_wo_p[i], 0.5)
        normalized_data = np.clip((normalized_data_pos + normalized_data_neg)*1.2, 0, 255)
        save_heatmap(output_dir, subfolder, f'Smoothgrad_cnn{channel_name_pos}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'Smoothgrad_{channel_name_pos}_heatmap.png'))

def apply_integratedgradients(images, model, subfolder, image_wo_p):
    '''
    Aplica Integrates Gradients al modelo

    Parámetros
    ----------
    images : tensor
        Tensor que contiene las imágenes de entrada (Vistas X,Y,Z)
    model : torch.nn.Module
        Modelo MVCNN preentrenado.
    subfolder : str
        Nombre del subdirectorio donde se guardarán las imágenes resultantes.
    image_wo_p : list
        Lista de imágenes originales del hueso para la superposición del mapa de calor.
    '''
    ig = IntegratedGradients(model)

    attribution, delta = ig.attribute(inputs= images, return_convergence_delta=True,n_steps=200)
        
    attribution = attribution.squeeze()

    channels = ['X', 'Y', 'Z']
    channel_to_index = {'X': 0, 'Y': 1, 'Z': 2}

    attributions = {channel: get_attribution_channel(attribution, idx)[1] for idx, channel in enumerate(channels)}

    attributions_pos = {}  # Diccionario para almacenar las atribuciones positivas
    attributions_neg = {}  # Diccionario para almacenar las atribuciones negativas
    
    for idx, channel in enumerate(channels):
        attr_channel = get_attribution_channel(attribution, idx)[1]
        pos, neg = separate_attributions(attr_channel)
        attributions_pos[channel] = pos
        attributions_neg[channel] = neg

    for (channel_name_pos, channel_data_pos), (channel_name_neg, channel_data_neg)  in zip(attributions_pos.items(),attributions_neg.items()):

        i = channel_to_index[channel_name_pos]

        normalized_data_pos, normalized_data_neg = normalized_2(channel_data_pos, channel_data_neg)
        heatmap_img = overlay_heatmap_on_image(normalized_data_pos, normalized_data_neg, images_wo_p[i], 0.5)
        normalized_data = np.clip((normalized_data_pos + normalized_data_neg)*1.2, 0, 255)
        save_heatmap(output_dir, subfolder, f'IntegratedGradients_cnn{channel_name_pos}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'IntegratedGradients_{channel_name_pos}_heatmap.png'))

def apply_gradfr(images, model, is_panorama, is_explainable, subfolder, image_wo_p):
    '''
    Aplica GradfR (Gradients for Regression) al modelo

    Parámetros
    ----------
    images : tensor
        Tensor que contiene las imágenes de entrada (Vistas X,Y,Z)
    model : torch.nn.Module
        MVCNN preentrenado
    is_panorama : bool
    is_explaianable: bool
        Dependiendo, del valor de ambas se selecciona una capa u otra.
    subfolder : str
        Nombre del subdirectorio donde se guardarán las imágenes resultantes.
    image_wo_p : list
        Lista de imágenes originales del hueso para la superposición del mapa de calor.
    '''
    def get_gradfr(images, target_layer, model):
    
        model.zero_grad()
        grad_cam = LayerGradCam(model, target_layer)
        
        attributions = grad_cam.attribute(images, relu_attributions= False)

        attributions = torch.abs(attributions)

        return attributions

    cnn_names = ["cnnX", "cnnY", "cnnZ"]
    for i, cnn in enumerate(cnn_names):
        if is_explainable:
            if is_panorama:
                target_layer = model.__getattr__(cnn)[-10]
            else:
                target_layer = model.__getattr__(cnn)[0][-1][-1].conv3
        else:
            if is_panorama:
                target_layer = model.__getattr__(cnn)[-12]
            else:
                target_layer = model.__getattr__(cnn)[0][-1][-1].conv3

        attributions = get_gradfr(images, target_layer, model)
        
        attributions = attributions.squeeze(0, 1).detach().cpu().numpy()
        
        normalized_data = normalized(attributions)
        heatmap_img = overlay_heatmap_on_image_gradfr(normalized_data, image_wo_p[i], 0.4)
        save_heatmap(output_dir, subfolder, f'GradCam_{cnn}.png', normalized_data)
        heatmap_img.save(os.path.join(output_dir, subfolder,f'GradCam_{cnn}_heatmap.png'))

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError(f"Se esperaba 'True' o 'False', pero se recibió: {s}")

if len(sys.argv) != 4:
    print("Uso: explanaibility.py [PANORAMACNN:True/False] [Explainable:True/False] [entrenamiento]")
    sys.exit(1)

PANORAMACNN = str_to_bool(sys.argv[1])
Explainable = str_to_bool(sys.argv[2])
entrenamiento = sys.argv[3]

# Cargar media y std desde fichero
with open('valores_mean_std.txt', 'r') as f:
    lines = f.readlines()
mean_train = [float(value) for value in lines[0].strip().split(',')]
std_train = [float(value) for value in lines[1].strip().split(',')]

# Configuración de las imágenes
image_size = (36,108)
image_dir = './data_img'
image_files = ['_0_panorama_ext_X.png', '_0_panorama_ext_Y.png', '_0_panorama_ext_Z.png']

# Configuración del modelo
if(Explainable):
    path = './entrenamientos/PanoramaCNNEx/' if PANORAMACNN else './entrenamientos/ResnetCNNEx/'
    output_dir = './heatmap' + ('PanoramaCNNEx/' if PANORAMACNN else 'ResnetCNNEx/')
    model = PanoramaCNNEx() if PANORAMACNN else ResnetCNNEx()
else:
    path = './entrenamientos/PanoramaCNN/' if PANORAMACNN else './entrenamientos/ResnetCNN/'
    output_dir = './heatmap' + ('PanoramaCNN/' if PANORAMACNN else 'ResnetCNN/')
    model = PanoramaCNN() if PANORAMACNN else ResnetCNN()

path += entrenamiento

# Configuración del dispositivo
print(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.load_state_dict(torch.load(path))  # Cargar el modelo
model.to(device)  # Mover el modelo al dispositivo

# Preprocesado de imágenes
preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((36,108),antialias=True),
            transforms.Normalize(mean=mean_train, std=std_train)
        ])

model = model.eval()

df = pd.read_csv('./test.csv', delimiter=',')
image_dir = df['Sample'].tolist()

new_df = df[['Sample', 'Edad']].copy()
predicted_ages_dict = {}
saliency_values_dict = {}

for subfolder in image_dir[0:1]:

    print(f"Procesando {subfolder}...")

    images_wo_p, images = zip(*[load_image(os.path.join('./data_img', subfolder, f'{subfolder}{file}'), preprocess) for file in image_files if os.path.exists(os.path.join('./data_img', subfolder, f'{subfolder}{file}'))])

    if images:
        images = torch.stack(images).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_age = model(images).item()
            if(Explainable):
                x_saliency_value = model.view_saliency_layer.output
            else:
                x_saliency_value =  torch.tensor([[0.0, 0.0, 0.0]], device='cuda:0')

        predicted_ages_dict[subfolder] = predicted_age
        x_saliency_list = x_saliency_value.tolist()
        saliency_values_dict[subfolder] = x_saliency_list[0]

        images.requires_grad = True

        apply_gradfr(images,model, PANORAMACNN, Explainable, subfolder, images_wo_p)

        apply_integratedgradients(images,model, subfolder, images_wo_p)

        apply_smoothgrad(images, model, subfolder, images_wo_p)

# Resultados a DataFrame
new_df['Edad_predicha'] = new_df['Sample'].map(predicted_ages_dict)
new_df['ponderacionX'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[0])
new_df['ponderacionY'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[1])
new_df['ponderacionZ'] = new_df['Sample'].map(lambda x: saliency_values_dict.get(x, [None, None, None])[2])

# Guardar resultados
new_df.to_csv('./predicciones.csv', index=False)
