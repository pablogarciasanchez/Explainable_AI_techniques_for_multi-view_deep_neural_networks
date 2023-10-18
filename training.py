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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import functions as fn
import sys
import random
import pickle
import datetime
sys.stdout.reconfigure(encoding='utf-8')

# Estructura [batch,vistas,rgb,altura,anchura]

def weighted_l1_loss(input, target, weight):
    # Calcula el error absoluto entre input y target sin reducirlo (obtiene un error por elemento)
    loss_per_element = torch.abs(input - target)
    
    # Calcula la media del error absoluto por muestra (a lo largo de la dimensión 1 si hay más dimensiones)
    mean_loss_per_sample = torch.mean(loss_per_element, dim=1, keepdim=True)
    
    # Pondera la pérdida media con el peso proporcionado
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
    # cargar datos
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
    # parámetros para detener el entrenamiento temprano
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
            
            # Solo accede a los pesos si estás en la fase de entrenamiento
            if phase == 'train':
                weights = data[2].to(device)
            else:
                weights = None

            optimizer.zero_grad()
            
            outputs = model(features)
            labels = labels.view(-1, 1)
            
            # Si hay pesos, úsalos para calcular la pérdida; de lo contrario, usa la pérdida regular
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
    for epoch in range(num_epochs + fine_tune_epochs):
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
                break

        # Comprobar si es el momento de comenzar con el ajuste fino
        if epoch == num_epochs - 1:
            print('Starting fine-tuning...')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=lr/100)

    print('Training complete')

    # Asumiendo que ya tienes training_mae_list y validation_mae_list definidos
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

def main():
    #PARAMETROS:
    PANORAMACNN = False
    Explainable = True

    '''
    # Establecer la semilla para PyTorch (para tensores y operaciones relacionadas con PyTorch)
    seed = 1234
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Establecer la semilla para NumPy (para operaciones relacionadas con NumPy)
    np.random.seed(seed)

    # Establecer la semilla para random (para otras operaciones aleatorias)
    random.seed(seed)
    '''

    # Definir las medias y desviaciones estándar de los datos de entrenamiento
    mean_train = [0.09538473933935165, 0.06967461854219437, 0.04578746110200882]
    std_train = [0.20621216297149658, 0.15044525265693665, 0.088300921022892]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(Explainable):
        path = './entrenamientos/PanoramaCNNEx/' if PANORAMACNN else './entrenamientos/ResnetCNNEx/'
    else:
        path = './entrenamientos/PanoramaCNN/' if PANORAMACNN else './entrenamientos/ResnetCNN/'
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
        
        #summary(model, input_size=(32, 3, 3, 108, 36))
        dataloaders = create_dataloaders({'train': 32, 'val': 32, 'test': 1}, mean_train, std_train)
        model = train_model(model, dataloaders, criterion, optimizer, device, path, num_epochs=100, fine_tune_epochs = 0, lr = lr)
    else: # Para el caso de ResNet
        lr = 0.001
        if(Explainable):
            model, criterion, optimizer = create_modelEx(device, PANORAMACNN, path, lr)
        else:
            model, criterion, optimizer = create_model(device, PANORAMACNN, path, lr)
        
        #summary(model, input_size=(1, 3, 3, 108, 36))
        dataloaders = create_dataloaders({'train': 32, 'val': 32, 'test': 1},mean_train,std_train)
        model = train_model(model, dataloaders, criterion, optimizer, device, path, num_epochs=100, fine_tune_epochs=100, lr=lr)

    model.load_state_dict(torch.load(path))
    true, pred = test_model(model, dataloaders['test'], device)
    fn.show_stats(true,pred)

if __name__ == "__main__":

    main()