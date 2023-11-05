import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from utils import PanoramaCNN
from utils import ResnetCNN
from utils import PanoramaCNNEx
from utils import ResnetCNNEx
from utils import CustomImageDataset
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import sys
import pickle
import datetime
import os
sys.stdout.reconfigure(encoding='utf-8')


def weighted_l1_loss(input, target, weight):
    
    loss_per_element = torch.abs(input - target)
    
    mean_loss_per_sample = torch.mean(loss_per_element, dim=1, keepdim=True)
    
    weighted_loss = mean_loss_per_sample * weight
    
    return weighted_loss

def validate(model, val_dataloader, device, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                features, labels = data[0].to(device), data[1].to(device)
                outputs = model(features)
                labels = labels.view(-1, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(val_dataloader)

def create_model(device, is_panorama, path, lr):
    if is_panorama:
        model = PanoramaCNN().to(device)
    else:
        model = ResnetCNN(trainable=False).to(device)
    
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer

def create_modelEx(device, is_panorama, path, lr):
    model = PanoramaCNNEx().to(device)
    if is_panorama:
        model = PanoramaCNNEx().to(device)
    else:
        model = ResnetCNNEx(trainable=False).to(device)
    
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer

def create_dataloaders(batch_sizes,mean_train,std_train):
    # Cargar datos
    print("Cargando conjuntos de datos...")
    training_data = CustomImageDataset("train.csv", "data_img", mean_train, std_train, True)
    print("Cargado Conjunto de entrenamiento")
    validation_data = CustomImageDataset("validation.csv", "data_img", mean_train, std_train, False)
    print("Cargado Conjunto de validacion")
    test_data = CustomImageDataset("test.csv","data_img", mean_train, std_train, False)
    print("Cargado Conjunto de test")

    dataloaders = {
        'train': DataLoader(training_data, batch_size=batch_sizes['train'], shuffle=True),
        'val': DataLoader(validation_data, batch_size=batch_sizes['val'], shuffle=True),
        'test': DataLoader(test_data, batch_size=batch_sizes['test'], shuffle=False)
    }
    return dataloaders

def plot_mae_from_file(filename, path):
    # Cargar las listas desde el archivo
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        training_mae_list = data['training_mae']
        validation_mae_list = data['validation_mae']

    # Generar el gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(training_mae_list, label='Entrenamiento MAE', color='blue')
    plt.plot(validation_mae_list, label='Validacion MAE', color='red')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.title('MAE Entrenamiento vs Validación')
    plt.legend()
    plt.grid(True)
    plt.savefig(path + "_plot.png")

def train_model(model, dataloaders, criterion, optimizer, device, path, num_epochs=25, fine_tune_epochs=25, lr=0.001):
    
    # Parámetros para detener el entrenamiento temprano
    n_epochs_stop = 20
    epochs_no_improve = 0
    min_val_loss = np.Inf

    training_mae_list = []
    validation_mae_list = []

    # Función para el entrenamiento y validación
    def run_epoch(phase):

        running_loss = 0.0
        mae_train_loss = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        # iterar sobre datos
        for i, data in enumerate(dataloaders[phase], 0): # 'phase' es 'train' o 'val'
            features, labels = data[0].to(device), data[1].to(device)
            
            if phase == 'train':
                weights = data[2].to(device)
            else:
                weights = None

            optimizer.zero_grad()
            
            outputs = model(features)
            labels = labels.view(-1, 1)
            
            if weights is not None:
                loss = weighted_l1_loss(outputs, labels, weights).mean()
                mae_loss = criterion(outputs, labels).mean()
            else:
                loss = criterion(outputs, labels).mean()

            if phase == 'train':
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()

            if phase == 'train':
                mae_train_loss += mae_loss.item()

        return running_loss / len(dataloaders[phase]), mae_train_loss / len(dataloaders[phase])

    # Durante el entrenamiento, registrar la pérdida de entrenamiento y validación
    epoch = 0
    while epoch < (num_epochs + fine_tune_epochs):
        print(f"Epoch {epoch}/{num_epochs + fine_tune_epochs - 1}")
        print('-' * 10)

        # Entrenar el modelo
        train_loss, mae_train_loss = run_epoch('train')
        training_mae_list.append(mae_train_loss)

        # Validar el modelo
        val_loss, _ = run_epoch('val')
        validation_mae_list.append(val_loss)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Lógica para detener el entrenamiento temprano
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), path)
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                if(epoch < num_epochs - 1):
                    epoch = num_epochs - 1
                    epochs_no_improve = 0
                else:
                    break

        # Comprobar si es el momento de comenzar con el ajuste fino
        if epoch == num_epochs - 1:
            print('Starting fine-tuning...')
            epochs_no_improve = 0
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=lr/100)
        epoch += 1

    print('Training complete')

    with open(path + 'mae_data.pkl', 'wb') as f:
        pickle.dump({
            'training_mae': training_mae_list,
            'validation_mae': validation_mae_list
        }, f)

    plot_mae_from_file(path + 'mae_data.pkl', path)

    return model

def test_model(model, dataloader, device):
    model.eval()
    true = []
    pred = []
    # calcular el error total y el error cuadrático medio en el conjunto de prueba
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            features, labels = data[0].to(device), data[1].to(device)
            outputs = model(features).view(-1)
            true.extend(labels.cpu().numpy().flatten().tolist())
            pred.extend(outputs.cpu().numpy().flatten().tolist())
    return np.array(true), np.array(pred)

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError(f"Se esperaba 'True' o 'False', pero se recibió: {s}")

def main(PANORAMACNN, Explainable):
    
    modelo_str = 'Modelo: '
    
    if PANORAMACNN:
        modelo_str += 'Panorama-CNN'
    else:
        modelo_str += 'Resnet-CNN'

    if Explainable:
        modelo_str += 'Ex'

    with open('valores_mean_std.txt', 'r') as f:
        lines = f.readlines()

    mean_train = [float(value) for value in lines[0].strip().split(',')]
    std_train = [float(value) for value in lines[1].strip().split(',')]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(Explainable):
        path = './entrenamientos/PanoramaCNNEx/' if PANORAMACNN else './entrenamientos/ResnetCNNEx/'
    else:
        path = './entrenamientos/PanoramaCNN/' if PANORAMACNN else './entrenamientos/ResnetCNN/'
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Obtener la fecha y hora actuales
    current_datetime = datetime.datetime.now()

    # Formatear la fecha y hora para agregar al nombre del archivo
    formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

    # Añadir "entrenamiento" y la fecha y hora al path
    path += f'entrenamiento_{formatted_datetime}'

    if PANORAMACNN:
        lr = 0.001
        if(Explainable):
            model, criterion, optimizer = create_modelEx(device, PANORAMACNN, path, lr)
        else:
            model, criterion, optimizer = create_model(device, PANORAMACNN, path, lr)
        
        dataloaders = create_dataloaders({'train': 32, 'val': 32, 'test': 1}, mean_train, std_train)
        model = train_model(model, dataloaders, criterion, optimizer, device, path, num_epochs=100, fine_tune_epochs = 0, lr = lr)
    else:
        lr = 0.001
        if(Explainable):
            model, criterion, optimizer = create_modelEx(device, PANORAMACNN, path, lr)
        else:
            model, criterion, optimizer = create_model(device, PANORAMACNN, path, lr)
        
        dataloaders = create_dataloaders({'train': 32, 'val': 32, 'test': 1},mean_train,std_train)
        model = train_model(model, dataloaders, criterion, optimizer, device, path, num_epochs=100, fine_tune_epochs=100, lr=lr)

    model.load_state_dict(torch.load(path))
    true, pred = test_model(model, dataloaders['test'], device)
    fn.show_stats(true,pred)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Uso: training.py [PANORAMACNN] [Explainable]")
        sys.exit(1)

    PANORAMACNN_bool = str_to_bool(sys.argv[1])
    Explainable_bool = str_to_bool(sys.argv[2])
    
    main(PANORAMACNN_bool, Explainable_bool)
