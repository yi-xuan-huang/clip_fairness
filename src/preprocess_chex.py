import pandas as pd
import re
import json
import os
from tqdm import tqdm



# get split split
patients = pd.read_csv('../data/raw/chexpert_patients.csv')
patients.columns = [c.strip().lower() for c in patients.columns]

patients['patient'] = patients['patient'].str.strip()
print(patients.head())
print(patients['primary_race'].unique())
print(patients['ethnicity'].unique())


def categorize_race(row):
    race = row['primary_race'].strip().lower()
    ethnicity = str(row['ethnicity']).strip().lower()

    if ethnicity in ['hispanic/latino', 'hispanic']:
        return 'HISPANIC'
    elif 'white' in race or 'caucasian' in race:
        return 'WHITE'
    elif 'asian' in race or 'pacific islander' in race or 'hawaiian' in race:
        return 'ASIAN'
    elif 'black' in race or 'african american' in race:
        return 'BLACK'
    elif 'indian' in race or 'native' in race:
        return 'NATIVE AMERICAN'
    else:
        return 'OTHER'
    
patients['race'] = patients.apply(categorize_race, axis=1)
print(patients['race'].value_counts())

patients['gender'] = patients['gender'].str.strip().str[0]
print(patients.head())

gt = pd.read_csv('../data/raw/chexpert_test_groundtruth.csv')
gt['patient'] = gt['Study'].str.extract(r'(patient\d+)/')

merged = pd.merge(patients, gt, how='inner', on = 'patient')
print(merged.shape)

new_df = {
    'patient': [],
    'image': [],
    'view':[]
}

for i, row in merged.iterrows():
    files = os.listdir(os.path.join('../data/raw', row['Study']))
    for image in files:
        new_df['patient'].append(row['patient'])
        new_df['image'].append(os.path.join(row['Study'],image))
        view = image.split('_')[1].split('.')[0]
        if view.strip().lower() == 'frontal':
            view = 'PA'
        elif view.strip().lower() == 'lateral':
            view = 'LATERAL'
        else:  
            view = 'OTHER'
        new_df['view'].append(view)

merged= pd.merge(pd.DataFrame(new_df), merged, on = 'patient')

merged['age'] = pd.cut(merged['age_at_cxr'], bins=[0, 40, 60, 80, 999], 
                       labels= ['18-40', '41-60', '61-80', '81+'], right=False)

merged.to_csv('../data/chexpert_final.csv', index=False)
data_dict = [{
    'image': row['image'],
    'view': row['view'],
    'race': row['race'],
    'gender': row['gender'],
    'age': row['age'],
    **{label: row[label] for label in [
        "Atelectasis", "Cardiomegaly", "Consolidation", 
        "Edema", "Enlarged Cardiomediastinum", "Fracture", 
        "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
        "Pneumothorax", "Support Devices"]}
     }
    for index, row in merged.iterrows()
]

with open(f"../data/chexpert.json", 'w') as f:
    json.dump(data_dict,f, indent=4)

# n_bootstrap = 100
# bootstrapped = []
# # Generate bootstrap samples
# for _ in range(n_bootstrap):
#     df = merged.sample(n=len(merged), replace=True)
#     bootstrapped.append(df)

# bootstrapped = pd.concat(bootstrapped, ignore_index=True)
# bootstrapped.to_csv('../data/chexpert_bs.csv',index=False)

# data_dict = [{
#     'image': row['image'],
#     'view': row['view'],
#     'race': row['race'],
#     'gender': row['gender'],
#     'age': row['age'],
#     **{label: row[label] for label in [
#         "Atelectasis", "Cardiomegaly", "Consolidation", 
#         "Edema", "Enlarged Cardiomediastinum", "Fracture", 
#         "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
#         "Pneumothorax", "Support Devices"]}
#      }
#     for index, row in bootstrapped.iterrows()
# ]

# with open(f"../data/chexpert_bs.json", 'w') as f:
#     json.dump(data_dict,f, indent=4)