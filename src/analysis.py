import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score


def main():
    section = 'full'
    # section = 'impr'
    data_path = "/shared/beamTeam/yhuang/data/PA_LATERAL"

    with open(f"{data_path}/{section}/res_train.json", 'r') as f:  
        mimic_train_dict = json.load(f)

    with open(f"{data_path}/{section}/res_val.json", 'r') as f:  
        mimic_val_dict = json.load(f)

    with open(f"{data_path}/{section}/chexpert/res.json", 'r') as f:  
        chexpert_dict = json.load(f)

    df_mimic = pd.read_csv(f"{data_path}/{section}/final_train.csv") 
    df_chexpert= pd.read_csv(f"{data_path}/{section}/chexpert/chexpert_final.csv") 
    df_chexpert['age'] =  pd.cut(df_chexpert['age_at_cxr'], 
                                bins=[0, 40, 60, 80, 999], 
                                labels= ['18-40', '41-60', '61-80', '81+'], right=False)
    
    df_chexpert['names'] = df_chexpert['image']
    df_chexpert = df_chexpert[['names', 'gender', 'race', 'age']]
    print(f"chexpert columns {df_chexpert.columns}")

    # heads = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
    #         "Fracture", "Lung Lesion",  "Pleural Effusion", "Pneumonia", "Pneumothorax", "Support Devices"]
    heads = ["Cardiomegaly", "Consolidation", "Edema",  
             "Pleural Effusion" ]
    all_res = {}

    for head in heads:
        mimic_train_res = mimic_train_dict[head]
        mimic_train_df = pd.DataFrame(mimic_train_res)

        mimic_val_res = mimic_val_dict[head]
        mimic_val_df = pd.DataFrame(mimic_val_res)

        mimic_res = pd.concat([mimic_train_df, mimic_val_df])
        mimic_res[['p', 'subject_id', 'study_id', 'dicom_id']] = mimic_res['names'].str.split('/', expand=True)
        mimic_res['subject_id'] = mimic_res['subject_id'].str.slice(1)
        mimic_res['study_id'] = mimic_res['study_id'].str.slice(1)
        mimic_res['dicom_id'] = mimic_res['dicom_id'].str.slice(0, -4)
        df_mimic['subject_id'] = df_mimic['subject_id'].astype(str)
        df_mimic['study_id'] = df_mimic['study_id'].astype(str)
        df_mimic['dicom_id'] = df_mimic['dicom_id'].astype(str)

        # mimic_res = mimic_res[['subject_id', 'study_id', 'dicom_id', 'race', 'gender', 'anchor_age', head]].copy()
        mimic_merged= pd.merge(df_mimic, mimic_res, how='inner', on = ['subject_id', 'study_id', 'dicom_id'])
        print(mimic_merged.shape)
        mimic_merged.to_csv(f"/shared/beamTeam/yhuang/data/PA_LATERAL/{section}/results/res_{head}.csv", index=False)
        print(mimic_merged.columns)
        mimic_merged['age'] = pd.cut(mimic_merged['anchor_age'], 
                                     bins=[0, 40, 60, 80, 999], 
                                     labels= ['18-40', '41-60', '61-80', '81+'], right=False)
        mimic_merged = mimic_merged[[ 'names', 'age', 'gender', 'race', 'labels',  'scores', 'preds']]
        print(f"mimic columns {mimic_merged.columns}")

        chexpert_res = chexpert_dict[head]
        chexpert_res_df = pd.DataFrame(chexpert_res)
        chexpert_merged= pd.merge(df_chexpert, chexpert_res_df, how='inner', on='names')
        chexpert_merged= chexpert_merged[[ 'names', 'age', 'gender', 'race', 'labels',  'scores', 'preds']]

        stacked = pd.concat([mimic_merged, chexpert_merged])
        stacked.to_csv(f"{data_path}/{section}/results/stacked.csv")
        print(f"stacked {stacked.shape}")

        res = fairness(stacked, categories=['race', 'gender', 'age'])
        print(res)
        all_res[head] = res
    
    with open(f"{data_path}/{section}/results/fairness_all.json", 'w') as f:
        json.dump(all_res, f, indent=4)

def fairness(res_data, categories=['gender', 'race', 'age']):
    data = res_data.dropna()
    res = {
        'statistical': {},
        'ppv': {},
        'fpr': {},
        'fnr': {},
        'accuracy': {},
        'auc': {}
    }
    

    for cat in categories:
      
        if cat == 'gender':
            groups = ['F', 'M']
        elif cat == 'race':
            groups = ['WHITE', 'BLACK','HISPANIC', 'ASIAN']
        else:
            groups = ['18-40', '41-60','61-80','81+']
        # groups = data[cat].unique()

        for group in groups:

            if group != 'NaN':
                sub = data[data[cat]==group]

                total = sub.shape[0]
                t = (sub['labels']==1.0).sum()
                f = (sub['labels']==0.0).sum()

                p = (sub['preds']==1.0).sum()
                n = (sub['preds']==0.0).sum()

                tp = ((sub['preds'] == 1.0) & (sub['labels'] == 1.0)).sum()
                fp = ((sub['preds'] == 1.0) & (sub['labels'] == 0.0)).sum()
                tn = ((sub['preds'] == 0.0) & (sub['labels'] == 0.0)).sum()
                fn = ((sub['preds'] == 0.0) & (sub['labels'] == 1.0)).sum()

                # statistical parity
                res['statistical'][group] = np.round(p/total,3)
                # predictive parity
                if tp+fp>0: 
                    res['ppv'][group] = np.round(tp/(tp+fp),3)
                else:
                    res['ppv'][group] = None
                # FPR
                if fp+tn>0:
                    res['fpr'][group] = np.round(fp/(fp+tn),3)
                else:
                    res['fpr'][group] = None
                # FNR
                if fn+tp>0: 
                    res['fnr'][group] = np.round(fn/(fn+tp),3)
                else:
                    res['fnr'][group] = None
                
                # Accuracy
                if total > 0:
                    res['accuracy'][group] = np.round((tp + tn) / total, 3)
                else:
                    res['accuracy'][group] = None
                # AUC
                if np.unique(sub['labels']).size > 1:  # Check if there's more than one unique label value
                    res['auc'][group] = np.round(roc_auc_score(sub['labels'], sub['scores']), 3)
                else:
                    res['auc'][group] = None

    return res



if __name__ == '__main__':
    main()