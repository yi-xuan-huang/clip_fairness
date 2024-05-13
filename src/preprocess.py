import pandas as pd
import numpy as np
import re
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


SPLIT = 'train'
KEEP_VIEWS = ['PA', 'LATERAL']
# KEEP_VIEWS = ['PA']
# SECTION = "full_report"
# SECTION = "impr"
# SECTION = "find_impr"
SECTION = "demo_impr"

# get split split
images = pd.read_csv('../data/raw/mimic-cxr-2.0.0-split.csv')
images = images[images['split'] == SPLIT]
images['report_id'] = 's'+images['study_id'].astype(str)
print(images.columns)
print(f"data shape {images.shape}")

# get report
print("Attaching reports")
for i, row in tqdm(images.iterrows(), total=images.shape[0]):
    subject_id = row['subject_id']
    study_id = row['study_id']
    p = 'p' + str(subject_id)[:2] 
    report_path = f'../data/raw/reports/{p}/p{subject_id}/s{study_id}.txt'
    with open(report_path, 'r') as f:
        text = f.read()
    text = text.replace('\n', ' ')
    images.loc[i, 'report_text'] = text
    images.loc[i, 'p'] =p 

print(images.head())    
print(f"Images with reports {images.shape}")
print(images.columns)

# demographic data
admissions = pd.read_csv("../data/raw/admissions.csv")
# Function to get the first non-missing value in a series
def first_non_missing(series):
    return series.dropna().iloc[0] if not series.dropna().empty else None
# Group by 'subject_id' and apply the function to the desired columns
demo = admissions.groupby('subject_id').agg({
    'insurance': first_non_missing,
    'race': first_non_missing
}).reset_index()
print(f"Demographics from admission file {demo.shape}")
sex_age = pd.read_csv("../data/raw/patients.csv")
demo = pd.merge(demo, sex_age, on='subject_id', how='inner')
print(f"Demographics after merging with sex and age{demo.shape}")

demo['subject_id'] = pd.to_numeric(demo['subject_id'], errors='coerce')
demo.to_csv(f'../data/demo_{SPLIT}.csv', index=False)

if SPLIT=='test':
    study_demo = pd.merge(images, demo, on='subject_id', how='inner')
else:
    study_demo = pd.merge(images, demo, on='subject_id', how='inner')

race_mapping = {
    'BLACK/AFRICAN AMERICAN': 'BLACK',
    'BLACK/CAPE VERDEAN': 'BLACK',
    'BLACK/CARIBBEAN ISLAND': 'BLACK',
    'WHITE': 'WHITE',
    'WHITE - EASTERN EUROPEAN': 'WHITE',
    'WHITE - RUSSIAN': 'WHITE',
    'WHITE - OTHER EUROPEAN': 'WHITE',
    'ASIAN - ASIAN INDIAN': 'ASIAN',
    'ASIAN - KOREAN': 'ASIAN',
    'ASIAN - CHINESE': 'ASIAN',
    'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
    'ASIAN': 'ASIAN',
    'AMERICAN INDIAN/ALASKA NATIVE': 'AMERICAN INDIAN/ALASKA NATIVE',
    'HISPANIC OR LATINO': 'HISPANIC',
    'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC',
    'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
    'PORTUGUESE': 'OTHER',
    'OTHER': 'OTHER',
    'UNABLE TO OBTAIN': 'OTHER',
    'UNKNOWN': 'OTHER',
    'MULTIPLE RACE/ETHNICITY': 'OTHER'
}

# Apply the mapping
study_demo['race'] =study_demo['race'].map(race_mapping)
print(study_demo['race'].unique())

print(f"studies with demographics {study_demo.shape}")

study_demo.to_csv(f'../data/study_demo_{SPLIT}.csv', index=False)

labels = pd.read_csv('../data/raw/mimic-cxr-2.0.0-chexpert.csv')
print(f"CheXpert labels {labels.shape}")

meta = pd.read_csv('../data/raw/mimic-cxr-2.0.0-metadata.csv')
print(f"Meta data {meta.shape}")

meta = meta.loc[meta['ViewPosition'].isin(KEEP_VIEWS), ['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
print(f"Meta data after keeping views{KEEP_VIEWS} {meta.shape}")

final= pd.merge(study_demo, labels, how='inner', on=['subject_id', 'study_id'])
final = pd.merge(final, meta, how='inner', on = ['subject_id', 'study_id', 'dicom_id'])

print(f"final with demographics and labels {final.shape}")
# final['report_text'] = final['report_text'].str.lower().str.replace('[^\w\s]', '')

# final = pd.read_csv(f'../data/final_{SPLIT}.csv')
# print(final.shape)
def extract_findings_impression(text, impr_only):
    out = ""
    if impr_only:
        impr_match = re.search(r'IMPRESSIONS?\s*:(.*)', text, re.IGNORECASE|re.DOTALL)
        if impr_match:
            out = f"IMPRESSION: {impr_match.group(1).strip()}"
            return out
        return ""

    find_match = re.search(r'FINDINGS?\s*:(.*)', text, re.IGNORECASE|re.DOTALL)
    if find_match:
        out = f"FINDINGS: {find_match.group(1).strip()}"
        return out
    return ""

final['age'] = pd.cut(final['anchor_age'], 
                                     bins=[0, 40, 60, 80, 999], 
                                     labels= ['18-40', '41-60', '61-80', '81+'], right=False)

final['gender'] = np.where(final['gender'].str.strip() == 'F', 'Female', np.where(
    final['gender'].str.strip() == 'M', 'Male', ""))

if SECTION == 'find_impr':
    final['report_text'] = final['report_text'].apply(extract_findings_impression, impr_only=False)
    final = final.loc[final['report_text']!=""]
    print(f"after restricting to finding and impressions {final.shape}")
elif SECTION in ('impr', 'demo_impr'):
    final['report_text'] = final['report_text'].apply(extract_findings_impression, impr_only=True)
    final = final.loc[final['report_text']!=""]
    final['report_text'] = (final['race'].astype(str) + ', ' + final['gender'].astype(str) + ', ' + 'Age ' + 
                            final['age'].astype(str) + '.' + final['report_text'] )
    print(f"final after restricting to impression {final.shape}")

final_report = final.drop_duplicates(subset=['subject_id', 'study_id']).copy()
print(f"reports only {final_report.shape}")
# gender_cnt = 0
# for i, row in final_report.iterrows():
#     if (' man ' in row['report_text'].lower() or ' woman ' in row['report_text'].lower() 
#         or ' male ' in row['report_text'].lower() or ' female ' in row['report_text'].lower() or 
#         ' he ' in row['report_text'] or ' she ' in row['report_text']):
#         gender_cnt += 1
# print(gender_cnt)

final_report.to_csv(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/report_demo_{SPLIT}.csv", index=False)
final.to_csv(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/final_{SPLIT}.csv", index=False)

if SPLIT == 'train':
    # Shuffle the dataset to ensure randomness
    data_shuffled = final.sample(frac=1, random_state=42)

    # Get unique subject_ids
    unique_subject_ids = data_shuffled['subject_id'].unique()

    # Split the list of unique subject_ids into train and validation sets
    train_subject_ids, val_subject_ids = train_test_split(unique_subject_ids, test_size=0.1, random_state=42)

    # Filter the original dataset based on the split subject_id lists
    train_data = data_shuffled[data_shuffled['subject_id'].isin(train_subject_ids)]
    val_data = data_shuffled[data_shuffled['subject_id'].isin(val_subject_ids)]

    # Reset indices
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    print(train_data.shape)
    print(val_data.shape)

    train_dict = [
        {
            "image": f"{row['p']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.png",
            "text": f"View: {row['ViewPosition']}. " +  row['report_text'].strip(),
            "view": row['ViewPosition'],
            **{label: row[label] for label in ["Atelectasis", "Cardiomegaly", "Consolidation", 
                                            "Edema", "Enlarged Cardiomediastinum", "Fracture", 
                                            "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
                                            "Pneumothorax", "Support Devices"]}
        }
        for index, row in train_data.iterrows()
    ]
    with open(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/train.json", 'w') as f:
        json.dump(train_dict,f, indent=4)

   
    val_dict = [
        {
            "image": f"{row['p']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.png",
            "text": f"View: {row['ViewPosition']}. " + row['report_text'].strip(),
            "view": row['ViewPosition'],
            **{label: row[label] for label in ["Atelectasis", "Cardiomegaly", "Consolidation", 
                                            "Edema", "Enlarged Cardiomediastinum", "Fracture", 
                                            "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
                                            "Pneumothorax", "Support Devices"]}
        }
        for index, row in val_data.iterrows()
    ]
    with open(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/val.json", 'w') as f:
        json.dump(val_dict,f, indent=4)
    
    categories = ['race','gender','insurance']
    for cat in categories:
        unique_values = val_data[cat].unique().tolist()
        for v in unique_values:
            os.makedirs(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/by_{cat}/{v}", exist_ok=True)
            sub = val_data[val_data[cat] == v]
            sub_dict = [ {
                "image": f"{row['p']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.png",
                "text": f"View: {row['ViewPosition']}. " + row['report_text'].strip(),
                **{label: row[label] for label in ["Atelectasis", "Cardiomegaly", "Consolidation", 
                                               "Edema", "Enlarged Cardiomediastinum", "Fracture", 
                                               "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
                                               "Pneumothorax", "Support Devices"]}
            } for index, row in sub.iterrows() ]

            with open(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/by_{cat}/{v}/val.json", 'w') as f:
                json.dump(sub_dict, f, indent=4)

elif SPLIT == 'test':
    data_dict = [
        {
            "image": f"{row['p']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.png",
            "text": f"View: {row['ViewPosition']}. " + row['report_text'].strip(),
            **{label: row[label] for label in ["Atelectasis", "Cardiomegaly", "Consolidation", 
                                               "Edema", "Enlarged Cardiomediastinum", "Fracture", 
                                               "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
                                               "Pneumothorax", "Support Devices"]}
        }
        for index, row in final.iterrows()
    ]
    with open(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/{SPLIT}.json", 'w') as f:
        json.dump(data_dict,f, indent=4)

    categories = ['race','gender','insurance']
    for cat in categories:
        unique_values = final[cat].unique().tolist()
        for v in unique_values:
            os.makedirs(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/by_{cat}/{v}", exist_ok=True)
            sub = final[final[cat] == v]
            sub_dict = [ {
                "image": f"{row['p']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
                "text": f"View: {row['ViewPosition']}. " + row['report_text'].strip(),
                **{label: row[label] for label in ["Atelectasis", "Cardiomegaly", "Consolidation", 
                                               "Edema", "Enlarged Cardiomediastinum", "Fracture", 
                                               "Lung Lesion",  "Pleural Effusion", "Pneumonia", 
                                               "Pneumothorax", "Support Devices"]}
            } for index, row in sub.iterrows() ]

            with open(f"../data/{'_'.join(KEEP_VIEWS)}/{SECTION}/by_{cat}/{v}/{SPLIT}.json", 'w') as f:
                json.dump(sub_dict, f, indent=4)

    # # Load the JSON data into a DataFrame
    # reports_df = pd.read_json("../data/all_reports1.json")
    # print(f"Reports shape: {reports_df.shape}")
    # reports_df.to_csv('../data/reports.csv', index=False)

    # # Preprocess the 'patient_id' to match the 'subject_id' format by removing the 'p' prefix
    # reports_df['patient_id'] = reports_df['patient_id'].str[1:]

    # # Rename the 'patient_id' column to 'subject_id' to match the other DataFrame
    # reports_df.rename(columns={'patient_id': 'subject_id'}, inplace=True)
    # reports_df['subject_id'] = pd.to_numeric(reports_df['subject_id'], errors='coerce')
    # # Merge the reports data with the race information using 'subject_id'