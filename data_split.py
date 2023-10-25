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
