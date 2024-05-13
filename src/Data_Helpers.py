import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import ast
import re
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import Report_Parser
from CXR_Datasets importMIMIC 


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    graphs = Batch.from_data_list(graphs)

    return {
        'image': [x['image'] for x in batch],
        'text': [x['text'] for x in batch],
    }


# def getLoaders(datasets, args=None, shuffle=True, bsize=32, drop_last =False, zeroworkers = False):
#     '''
#     Returns dataloaders for each dataset in datasets
#     '''
#     subset = datasets.keys()
#     if not zeroworkers:
#         num_work = min(os.cpu_count(), 16)
#         num_work = num_work if num_work > 1 else 0
#     else:
#         num_work = 0
#     batch_size = args.batch_size if args else bsize
#     prefetch_factor = 2
#     loaders = {}
#     for sub in subset:
#         loaders[sub] = DataLoader(datasets[sub], batch_size=batch_size, shuffle=shuffle, num_workers=num_work,
#                                   prefetch_factor=prefetch_factor, pin_memory=True, drop_last = drop_last, collate_fn=collate_fn)
#     return loaders

def get_loader(dataset, args=None, shuffle=False, bsize=32, drop_last =False, zeroworkers = False):
    '''
    Returns dataloaders for each dataset in datasets
    '''
    if not zeroworkers:
        num_work = min(os.cpu_count(), 10)
        num_work = num_work if num_work > 1 else 0
    else:
        num_work = 0
    batch_size = args.batch_size if args else bsize
    prefetch_factor = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_work, 
                        prefetch_factor=prefetch_factor, pin_memory=True, drop_last = drop_last, collate_fn=collate_fn)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_work, 
    #                     prefetch_factor=prefetch_factor, pin_memory=True, drop_last = drop_last)

    # graph_loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_work, 
    #                     prefetch_factor=prefetch_factor, pin_memory=True, drop_last = drop_last)
    # return loader, graph_loader
    return loader
# def getFilters(exp_path, overwrite = '', toprint=True): #return filters that were used to train an experiment, if possible
#     '''
#     return filters that were used to train an experiment, if possible
#     Looking for 'filters.txt' in the exp folder.
#     Alternatively, can overwrite the filters used
#     '''
#     try:
#         if overwrite == '':
#             return []
#         if overwrite is not '' and type(overwrite) == str:
#             if toprint:
#                 logging.info("Overwriting filters with " + overwrite)
#             return overwrite.split(",")
#         else:
#             txt_file = open(exp_path + '/filters.txt', "r")
#             file_content = txt_file.read()
#             content_list = file_content.split(",")
#             txt_file.close()
#             if toprint:
#                 logging.info("Using filter file with " + file_content)
#             return content_list
#     except:
#         if toprint:
#             logging.info("No filter file found, none applied.")
#         return []

def getImList(sr, group, fps, heads=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], filters = []):
    '''
        Returns the dataframe of samples (il) to use and the root directory for the data
    '''
    if sr == 'mimic_cxr':
        rd = fps['mimic_root_dir']
        il_labels = pd.read_csv(fps['mimic_chex_file'])
        il_meta = pd.read_csv(fps['mimic_meta_file']).loc[:, ['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
        il = pd.read_csv(fps['mimic_csv_file'])
        il = il.merge(il_labels, on=['subject_id', 'study_id'])
        il = il.merge(il_meta, on=['dicom_id','subject_id', 'study_id'])
        if 'frontal' in filters:
            il = il[np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA')]
        elif 'lateral' in filters:
            il = il[np.logical_not(np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA'))]

        # logging.info(il.columns)
        logging.info(il.shape)
        logging.info(np.unique(il['subject_id'].values).shape)
        il = getSplits(il, group)
        #logging.info(group, il.shape)
        il['pGroup'] = np.array(["p" + pg[:2] for pg in il['subject_id'].values.astype(str)])
        il['pName'] = np.array(["p" + pn for pn in il['subject_id'].values.astype(str)])
        il['sName'] = np.array(["s" + sn for sn in il['study_id'].values.astype(str)])
        if 'findings' in filters:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, findings_only=True), axis=1)
            il = il[il['text'] != ""]
        elif 'impression' in filters:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, impression_only=True), axis=1)
            il = il[il['text'] != ""]
            logging.info(il.shape)
        else:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, impression_only=True), axis=1)
            il['findings'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, findings_only=True), axis=1)
            il['impression'] = il['text']
            il = il[il['findings'] != ""]
            il = il[il['impression'] != ""]

    elif sr == 'chexpert':
        rd = fps['chexpert_root_dir']
        il_train = pd.read_csv(rd + 'CheXpert-v1.0-small/train.csv')
        il_train['patName'] = il_train['Path'].str.extract(r'train/(.*?)/study')
        test = pd.read_csv(rd + 'CheXpert-v1.0-small/valid.csv')
        if 'frontal' in filters:
            il_train = il_train[il_train['Frontal/Lateral'] == 'Frontal']
            test = test[test['Frontal/Lateral'] == 'Frontal']
        elif 'lateral' in filters:
            il_train = il_train[il_train['Frontal/Lateral'] == 'Lateral']
            test = test[test['Frontal/Lateral'] == 'Lateral']
        if group == 'test':
            il = test

        elif group != 'all':
            il = getSplits(il_train, group, 'chexpert', heads)
        else:
            il = pd.concat((il_train, test), axis=0)


    elif sr == 'covid-chestxray':
        rd = fps['covid_chestxray_csv_file']
        il = pd.read_csv(rd)
        il['Pneumonia'] = il['label'].str.contains('Pneumonia')
        il['No Finding'] = il['label'].str.contains('No Finding')
        il['covid19'] = il['label'].str.contains('covid19')
        il = il[il['label'].isin(heads)]
        if group == 'train' or group == 'val' or group == 'test':
            il = il[il['group'].str.contains(group)]
        if 'tiny' in group:
            il = il[::50]

    elif sr == 'padchest':
        rd = fps['padchest_root_dir']
        il = pd.read_csv(fps['padchest_file'], compression='gzip', header=0)
        il['im_path'] = rd + il['ImageDir'].astype(str) + '/' + il['ImageID']
        if 'frontal' in filters:
            il = il[il['Projection'].str.contains('AP|PA')]
        elif 'lateral' in filters:
            il = il[~il['Projection'].str.contains('AP|PA')]

        il = il[il['MethodLabel'] == 'Physician']
        if 'tiny' in group:
            il = il[::10]

        il = il.loc[:, ['im_path', 'Labels']]
        labellist = il.pop('Labels').values.astype(str)
        outlist = []
        for i, l in enumerate(labellist):
            try:
                outlist.append(ast.literal_eval(l))
            except:
                outlist.append([])

        mlb = MultiLabelBinarizer(sparse_output=False)
        toadd = pd.DataFrame(
            mlb.fit_transform(outlist),
            index=il.index,
            columns=mlb.classes_)

        high_importance = ['COPD signs', 'endotracheal tube', 'pleural effusion', 'pulmonary edema', 'heart insufficiency',
                           'pulmonary fibrosis', 'cardiomegaly', 'vascular redistribution', 'consolidation', 'hilar congestion',
                           'pulmonary mass', 'cavitation', 'alveolar pattern', 'calcified pleural thickening', 'lung metastasis',
                           'emphysema', 'interstitial pattern', 'costophrenic angle blunting','tuberculosis','atelectasis',
                           'reticular interstitial pattern','pneumonia','lobar atelectasis','normal', 'pleural thickening',
                           'reticulonodular interstitial pattern','infiltrates','hypoexpansion','hypoexpansion basal',
                           'humeral fracture','pneumothorax','multiple nodules','hyperinflated lung', 'bronchiectasis',
                           'adenopathy','mediastinal enlargement','laminar atelectasis','vertebral compression','rib fracture',
                           'tuberculosis sequelae', 'hilar enlargement', 'tracheal shift', 'mediastinal mass', 'central vascular redistribution',
                           'vertebral fracture','superior mediastinal enlargement','vascular hilar enlargement','nodule',
                           'air trapping','bullas', 'ground glass pattern', 'calcified adenopathy', 'minor fissure thickening',
                           'unchanged', 'clavicle fracture','pseudonodule','end on vessel']
        for h in high_importance:
            try:
                assert h in toadd.columns
            except:
                logging.info(h)
        toadd = toadd.loc[:, high_importance]
        il = il.join(toadd)


    return il, rd


def getSplits(df, group, sr='mimic_cxr', heads=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']):
    '''
    Returns the data for a specified group (train/val)
    '''
    if sr == 'mimic_cxr':
        if 'tiny' in group:
            df = splitDF(df, 'subject_id', 0.01)[1]
        elif 'med' in group:
            df = splitDF(df, 'subject_id', 0.3)[1]

        train, val = splitDF(df, 'subject_id', 0.1)  # train, val
        if 'train' in group:
            df = train
        elif 'val' in group or 'test' in group:
            df = val

    elif sr == 'chexpert':
        il_unseen, il_finetune = splitDF(df, 'patName', 0.1)
        il_finetune_train, il_finetune_val = splitDF(il_finetune, 'patName', 0.1)
        if 'train' in group:
            df = il_finetune_train
        elif 'val' in group:
            df = il_finetune_val
        elif 'candidates' in group or 'queries' in group:
            il = df
            il = il.drop_duplicates(subset=['patName'])
            temp = il.loc[:, heads]
            tempsum = (temp.values == 1).sum(axis=1) == 1
            unknownsum = (temp.values == -1).sum(axis=1) == 0
            both = np.logical_and(tempsum, unknownsum)
            uniquePos = il.iloc[both, :]
            uniquePosPer = [uniquePos[uniquePos[h] == 1] for h in heads]
            if 'candidates' in group:
                uniquePosLim = [u.iloc[:100, :] for u in uniquePosPer]
            else:
                uniquePosLim = [u.iloc[100:120, :] for u in uniquePosPer]
            df = pd.concat(uniquePosLim)
    return df


def splitDF(df, patientID, testsize=0.2):
    '''
    Splitting data with given test size with all data from a given patient in one group
    '''
    if testsize == 1.0:
        return df, df
    splitter = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state=1)
    split = splitter.split(df, groups=df[patientID])
    train_inds, valtest_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[valtest_inds]
    return train, test


def textProcess(text):
    parsed = Report_Parser.parse_report(text)
    if 'findings' in parsed:
        return parsed['findings']
    else:
        return "NO FINDINGS"