'''

import pandas as pd
import functions as fn

# Cargar el dataframe
df = pd.read_csv('pubis.csv')

# Crear una copia del dataframe para evitar cambiar el original
df_copy = df.copy()

# Mezclar el dataframe antes de la división para evitar sesgos
df_copy = df_copy.sample(frac=1, random_state=42)

# Divide el conjunto de datos basado en los individuos en lugar de las observaciones individuales
train_df, test_df = train_test_split(df_copy['Number'].drop_duplicates(), test_size=0.1, random_state=42)

# Obtén las filas del conjunto de datos original que corresponden a los individuos en los conjuntos de entrenamiento y prueba
train_df = df_copy[df_copy['Number'].isin(train_df)]
test_df = df_copy[df_copy['Number'].isin(test_df)]

# De lo anterior, dividir el conjunto de entrenamiento en entrenamiento (75%) y validación (25%)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

print(len(train_df))
print(len(test_df))
print(len(val_df))

fn.calculate_weights(train_df)

train_df.to_csv('./train.csv',index=False)
test_df.to_csv('./test.csv',index=False)
val_df.to_csv('./val.csv',index=False)
'''

import math
import pandas as pd
import functions as fn

sample_df = pd.read_csv('./pubis.csv')

age_list = sample_df['Edad'].to_list()
dic_ranges = dict.fromkeys([fn.bin(label,age_list) for label in age_list],0)
sample_range_list = []

for sample in sample_df.index:
    age = sample_df.at[sample,'Edad']
    age = int(age)
    for i,r in enumerate(dic_ranges.keys()):
        if r == fn.bin(age,age_list):
            dic_ranges[r] += 1
            sample_range_list.append(i)
            break

#print(dic_ranges)

sample_df.insert(2,'Range', sample_range_list, True)
#print(sample_df)

#print('\n[INFO] Calculando pesos...\n')

#print(age_list)

#print(sample_df)
train_val, test = fn.train_test_split(sample_df,0.9)

#print(train_val)
#print(test)
train, validation = fn.train_test_split(train_val,0.75)

fn.calculate_weights(train)

#print(train)
#print(validation)
#print(test)

train = train.drop(columns=["Range"])
train.to_csv('./train.csv',index=False)
validation = validation.drop(columns=["Range"])
validation.to_csv('./validation.csv',index=False)
test = test.drop(columns=["Range"])
test.to_csv('./test.csv',index=False)
train_val = train_val.drop(columns=["Range","Set"])
train_val.to_csv('./trainval.csv',index=False)

sample_df = sample_df.drop(columns=["Range","Set"])
sample_df.to_csv('./dataset.csv',index=False)