from utils import PanoramaCNNEx, ResnetCNNEx, ViewSaliencyLayer, ViewSaliencyFunction, SaliencyBasedPoolingLayer, SaliencyBasedPoolingFunction
from torchinfo import summary
import sys
import torch
from torch import nn
from torch.autograd import gradcheck
import numpy as np
import random
sys.stdout.reconfigure(encoding='utf-8')
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#model = ResnetCNNEx()

#summary(model, input_size=(1, 3, 3, 108, 36))

def plot_multiple_mae(filenames, labels, path):
    
    # Gráfica para el entrenamiento
    plt.figure(figsize=(12, 6))
    
    for filename, label in zip(filenames, labels):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            training_mae_list = data['training_mae']
            
        # Graficar las listas de MAE de entrenamiento
        plt.plot(training_mae_list, label=f'{label} Mae train', alpha=0.9)

    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.title('MAE Entrenamiento para 5 Entrenamientos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + "_training_combined_plot.png")

    # Gráfica para la validación
    plt.figure(figsize=(12, 6))

    colors = colors = plt.cm.tab10.colors[:len(filenames)]

    for filename, label, color in zip(filenames, labels, colors):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            validation_mae_list = data['validation_mae']

        # Graficar las listas de MAE de validación
        plt.plot(validation_mae_list, color=color, label=f'{label} Mae val', alpha=0.2)

        # Mostrar el valor mínimo obtenido hasta el momento
        min_values = np.minimum.accumulate(validation_mae_list)
        plt.plot(min_values, color=color,alpha=0.9, label=f'{label} Mae min val')  # Añadimos label aquí

    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.title('MAE Validación para 5 Entrenamientos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + "_validation_combined_plot.png")

# Uso de la función
path = './entrenamientos/PanoramaCNNEx/entrenamiento'
filenames = [path + '_231003_1746mae_data.pkl', path + '_231004_1329mae_data.pkl', path + '_231004_1416mae_data.pkl', path + '_231004_1633mae_data.pkl', path + '_231004_1752mae_data.pkl']  # Añade todos los nombres de archivos aquí
labels = ['1', '2', '3', '4', '5']  # Etiquetas correspondientes a cada entrenamiento
plot_multiple_mae(filenames, labels, 'combined_path')

'''
view_saliency_layer = ViewSaliencyLayer()

sp = SaliencyBasedPoolingLayer()

class ViewSaliencyModule(nn.Module):
    def forward(self, x):
        return ViewSaliencyFunction.apply(x)
    
class SaliencyBasedPoolingModule(nn.Module):
    def forward(self, output_relu, output_vs):
        return SaliencyBasedPoolingFunction.apply(output_relu, output_vs)
    

relu = torch.tensor([[[1.5,0,1],[1,0.5,0.8],[1.5,2,3],[2,1,2]]])
saliency = torch.tensor([[0.5,1.0]])
print("Dimension F: ", relu.shape, " ", relu)
#print("Dimension S: ", saliency.shape, saliency)

output = view_saliency_layer(relu)
print(output)

module = ViewSaliencyModule()

input_tensor = torch.randn(1, 10, 3, dtype=torch.float64, requires_grad=True)
module = ViewSaliencyModule().double()

#print("Tensor de prueba: ", input_tensor)
print("Dimension del tensor de prueba: (B, F, N)", input_tensor.shape)

test = gradcheck(func = module,inputs= (input_tensor,))
print('Probando la capa VS')
print(test)

input_tensor1 = torch.randn(1, 100, 3, dtype=torch.float64, requires_grad=True)
input_tensor2 = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
module = SaliencyBasedPoolingModule().double()

# Verificar el gradiente
test = gradcheck(module, (input_tensor1,input_tensor2, ))
print(test)
'''