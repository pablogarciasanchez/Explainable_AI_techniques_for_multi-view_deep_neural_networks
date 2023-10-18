import torch
from torch.utils.data import Dataset
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image

import os
import pandas as pd
import glob
import re

from torch.autograd import Function

### DEFINICION DATASET
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mean, std, with_weights=True):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.with_weights = with_weights

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((36,108),antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.image_triplets = []

        image_paths = glob.glob(os.path.join(self.img_dir, '*/*_panorama_ext_*.png'))
        image_files_grouped = {}
        for path in image_paths:
            base_path, file_name = os.path.split(path)
            sample_number = re.search(r'_(\d+)_panorama_ext_', file_name).group(1)
            if base_path not in image_files_grouped:
                image_files_grouped[base_path] = {}
            if sample_number not in image_files_grouped[base_path]:
                image_files_grouped[base_path][sample_number] = []
            image_files_grouped[base_path][sample_number].append(path)

        for _, row in self.img_labels.iterrows():
            base_img_path = os.path.join(self.img_dir, row[1])
            if base_img_path in image_files_grouped:
                for sample, paths in image_files_grouped[base_img_path].items():
                    image_files = sorted(paths)
                    if with_weights:
                        self.image_triplets.append((self.transform_images(image_files), row[2], row[3]))
                    else:
                        self.image_triplets.append((self.transform_images(image_files), row[2]))

    def transform_images(self, image_files):
        images = []
        for img_file in image_files:
            image = Image.open(img_file)
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        return images

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        if self.with_weights:
            images, label, weight = self.image_triplets[idx]
            return images, label, weight
        else:
            images, label = self.image_triplets[idx]
            return images, label

### DEFINICION DE LOS MODELOS
class PanoramaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnnX = self.make_cnn()
        self.cnnY = self.make_cnn()
        self.cnnZ = self.make_cnn()

    def make_cnn(self):

        model = nn.Sequential(

            # Primera capa
            nn.ZeroPad2d(2),
            nn.Conv2d(3, 64, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 256, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Tercera capa
            nn.ZeroPad2d(2),
            nn.Conv2d(256, 1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(71680,100),
            
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1),
            nn.ReLU()
        )

        return model

    def forward(self, x):

        # Pasar cada vista a través de su propia CNN
        xs = []
        xs.append(self.cnnX(x[:, 0]))
        xs.append(self.cnnY(x[:, 1]))
        xs.append(self.cnnZ(x[:, 2]))

        x = torch.stack(xs, dim=2)
        x = torch.mean(x, dim=2)
        return x

class ResnetCNN(nn.Module):
    def __init__(self, trainable=False):
        super(ResnetCNN, self).__init__()
        self.trainable = trainable

        self.cnnX = self.make_cnn()
        self.cnnY = self.make_cnn()
        self.cnnZ = self.make_cnn()
        

    def make_cnn(self):
        resnet = models.resnet50(pretrained=True)
        if not self.trainable:
            for param in resnet.parameters():
                param.requires_grad = False

        # Remove the last (classification) layer of ResNet-50
        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)

        # Add the new layers on top of the resnet
        model = nn.Sequential(
            self.resnet,
            nn.Flatten(),
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1),
            nn.ReLU()
        )
        return model

    def forward(self, x):
        xs = []
        xs.append(self.cnnX(x[:, 0]))
        xs.append(self.cnnY(x[:, 1]))
        xs.append(self.cnnZ(x[:, 2]))

        x = torch.stack(xs, dim=2)
        x = torch.mean(x, dim=2)
        return x
 
class ViewSaliencyFunction(Function):

    @staticmethod
    def forward(ctx, x):
        B, _, N = x.shape

        # Usamos broadcasting para calcular todas las distancias pairwise de una vez para todos los batches
        x_expanded = x.unsqueeze(2)  # Shape: [B, D, 1, N]
        x_tiled = x.unsqueeze(3)  # Shape: [B, D, N, 1]
        
        # Pairwise differences
        diffs = x_expanded - x_tiled  # Shape: [B, D, N, N]
        
        # Euclidean distances along the feature dimension (dim=1)
        D = torch.norm(diffs, dim=1) # Shape: [B, N, N]

        # No considerar la diagonal, la ponemos a 0
        diag_mask = torch.eye(N, device=x.device).unsqueeze(0).bool()
        D.masked_fill_(diag_mask, 0)

        # Sumar las distancias para calcular la relevancia
        S = D.sum(dim=2)  # Shape: [B, N]

        # Calcula la suma para cada fila (batch)
        sums = S.sum(dim=1, keepdim=True)  # Shape: [B, 1]

        # Normaliza saliency_stacked dividiendo cada fila por su suma
        R = S / (sums + 1e-10)  # Se añade 1e-10 para evitar la división por cero

        ctx.save_for_backward(x, diffs, S, sums)

        return R
    
    @staticmethod
    def backward(ctx, grad_output):
        x, diffs, S, sums = ctx.saved_tensors
        B, D, N = x.shape

        grad_sums = -(grad_output * S / (sums * sums)).sum(dim=1, keepdim=True)
        grad_saliency_stacked = grad_output / sums + grad_sums

        normed_diffs = diffs / (torch.norm(diffs, dim=1, keepdim=True) + 1e-10)
        
        expanded_grad_saliency_stacked = grad_saliency_stacked.unsqueeze(1).unsqueeze(2)
        
        grad_x_i = -(normed_diffs * expanded_grad_saliency_stacked).sum(dim=3)
        grad_x_j = (normed_diffs * expanded_grad_saliency_stacked).sum(dim=2)
        
        grad_x = grad_x_i + grad_x_j

        return grad_x, 

class ViewSaliencyLayer(nn.Module):
    def __init__(self):
        super(ViewSaliencyLayer, self).__init__()

    def forward(self, x):
        return ViewSaliencyFunction.apply(x)

class SaliencyBasedPoolingFunction(Function):

    @staticmethod
    def forward(ctx, output_relu, output_vs):
        # Asumiendo output_relu: (batch_size, feature_dim, N) y output_vs: (batch_size, N)
        ctx.save_for_backward(output_relu, output_vs)
        
        # Suma ponderada a lo largo de la dimensión N
        pooled_output = (output_relu * output_vs.unsqueeze(1)).sum(dim=-1)  # Resultado: (batch_size, feature_dim)

        return pooled_output

    @staticmethod
    def backward(ctx, grad_output):
        output_relu, output_vs = ctx.saved_tensors

        # Gradiente respecto a output_relu
        grad_input_relu = grad_output.unsqueeze(-1) * output_vs.unsqueeze(1)
        # Gradiente respecto a output_vs
        grad_input_vs = torch.bmm(output_relu.transpose(1,2), grad_output.unsqueeze(-1)).squeeze(-1)

        return grad_input_relu, grad_input_vs

class SaliencyBasedPoolingLayer(nn.Module):
    def __init__(self):
        super(SaliencyBasedPoolingLayer, self).__init__()

    def forward(self, output_relu, output_vs):
        return SaliencyBasedPoolingFunction.apply(output_relu, output_vs)
    
def hook_fn(module, input, output):
    # This will capture the output of the view_saliency_layer
    module.output = output
    
class PanoramaCNNEx(nn.Module):
    def __init__(self):
        super().__init__()

        # Definición de la CNN para cada vista
        self.cnnX = self.make_cnn()
        self.cnnY = self.make_cnn()
        self.cnnZ = self.make_cnn()
        
        self.post = self.post_relu()
        
        self.view_saliency_layer = ViewSaliencyLayer()
        self.view_saliency_layer.register_forward_hook(hook_fn)
        self.salience_based_pooling_layer = SaliencyBasedPoolingLayer()

    def make_cnn(self):

        model = nn.Sequential(

            # Primera capa
            nn.ZeroPad2d(2),
            nn.Conv2d(3, 64, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 256, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Tercera capa
            nn.ZeroPad2d(2),
            nn.Conv2d(256, 1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(71680, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1)

        )

        return model
    
    def post_relu(self):

        model = nn.Sequential(
            nn.Linear(100,1),
            nn.ReLU()
        )

        return model

    def forward(self, x):

        features_X = self.cnnX(x[:, 0])
        features_Y = self.cnnY(x[:, 1])
        features_Z = self.cnnZ(x[:, 2])
        
        x_intermediate_combined = torch.stack([features_X, features_Y, features_Z], dim=2)

        x_saliency = self.view_saliency_layer(x_intermediate_combined)

        pooling_weights = self.salience_based_pooling_layer(x_intermediate_combined, x_saliency)

        x = self.post(pooling_weights)

        return x

class ResnetCNNEx(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable
        
        self.cnnX = self.make_cnn()
        self.cnnY = self.make_cnn()
        self.cnnZ = self.make_cnn()

        self.post = self.post_relu()
        
        self.view_saliency_layer = ViewSaliencyLayer()
        self.view_saliency_layer.register_forward_hook(hook_fn)
        self.salience_based_pooling_layer = SaliencyBasedPoolingLayer()

    def make_cnn(self):
        resnet = models.resnet50(pretrained=True)
        if not self.trainable:
            for param in resnet.parameters():
                param.requires_grad = False

        # Remove the last (classification) layer of ResNet-50
        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)

        model = nn.Sequential(
            self.resnet,
            nn.Flatten(),
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        return model

    def post_relu(self):

        model = nn.Sequential(
            nn.Linear(100,1),
            nn.ReLU()
        )

        return model

    def forward(self, x):

        features_X = self.cnnX(x[:, 0])
        features_Y = self.cnnY(x[:, 1])
        features_Z = self.cnnZ(x[:, 2])
        
        x_intermediate_combined = torch.stack([features_X, features_Y, features_Z], dim=2)

        x_saliency = self.view_saliency_layer(x_intermediate_combined)

        pooling_weights = self.salience_based_pooling_layer(x_intermediate_combined, x_saliency)

        x = self.post(pooling_weights)

        return x
