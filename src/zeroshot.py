import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from CXR_Datasets import MIMIC_Labels
import CLIP_Embedding
# from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_prompts(heads):
    print(heads)

    pos_prompts = {
    'Atelectasis': [
        "Atelectasis is present.", 
        "Basilar opacity and volume loss is likely due to atelectasis.",
        "Mild basilar atelectasis is seen"],
    'Cardiomegaly': [
        "Cardiomegaly is present.", 
        "The heart shadow is enlarged.", 
        "The cardiac silhouette is enlarged.",
        "The heart size is enlarged"], 
    'Consolidation': ["Consolidation is present.", "Dense white area of right lung indicative of consolidation."],
    'Edema': ["Edema is present.", "Increased fluid in the alveolar wall indicates pulmonary edema."],
    "Enlarged Cardiomediastinum": ["Chest X-ray shows a widened mediastinum, suggestive of an Enlarged Cardiomediastinum.",
                                    "increased mediastinal width, indicative of an Enlarged Cardiomediastinum.", 
                                    "Cardiothoracic ratio exceeds normal limits, consistent with an Enlarged Cardiomediastinum."],
    'Fracture': ["a linear lucency or disruption in bone continuity, suggestive of a rib fracture", 
                 "cortical disruption and bony discontinuity consistent with a rib fracture",
                 "Evidence of rib fracture is observed", 
                 "a displaced or angulated rib segment, indicative of a fracture"],
    'Lung Lesion': ["shows a solitary pulmonary nodule, indicative of a lung lesion.", 
                    "Radiographic findings reveal a focal opacity or density in the lung parenchyma, suggestive of a lesion.", 
                    "a well-defined or ill-defined mass in the lung, consistent with a lesion.", 
                    "opacity or consolidation is observed on chest X-ray, indicating a possible lung lesion."],
    'Pleural Effusion': ["Pleural Effusion is present.", "Blunting of the costophrenic angles represents pleural effusions.", 
                         "The pleural space is filled with fluid.", "Layering pleural effusions are present."],
    'Pneumonia': ["Patchy or confluent opacities consistent with pneumonia.", 
                  "Airspace consolidation indicative of pneumonia.", 
                  "Area of increased opacity or density in the lung fields, suggestive of pneumonia.", 
                  "Opacity or consolidation with air bronchograms, characteristic of pneumonia.", 
                  "Lobar or multilobar infiltrates, consistent with pneumonia."],
    "Pneumothorax": ["Visible lung collapse on chest X-ray, indicative of pneumothorax.", 
                     "hyperlucent lung fields with absent lung markings, suggestive of pneumothorax.", 
                     "a visceral pleural line with absence of lung markings beyond, characteristic of pneumothorax.", 
                     "an air-fluid level at the pleural apex, indicative of pneumothorax.", 
                     "a visible demarcation between the lung edge and the chest wall, suggestive of pneumothorax." ], 
    "Support Devices": [ "Presence of endotracheal tube noted on chest X-ray, indicating respiratory support.", 
                        "Radiographic findings reveal a central venous catheter in situ, suggestive of hemodynamic support.", 
                        "X-ray demonstrates a nasogastric tube placement, indicative of nutritional support.", 
                        "Chest radiograph reveals a urinary catheter in place, suggestive of urinary support.", 
                        "Radiographic assessment shows a surgical drain in situ, indicative of wound drainage support." ]
    }

    neg_query = ["The lungs are clear.", "No abnormalities are present.", "The chest is normal.", 
             "No clinically signiffcant radiographic abnormalities.", "No radiographically visible abnormalities in the chest."]
    neg_prompts = {disease: [f'{neg}' for neg in neg_query] for disease in heads}
    neg_prompts['Cardiomegaly'].extend([
        "The heart size is normal", 
        "The cardiac size is normal", 
        "Normal hilar and  mediastinal contours.",
        "Normal appearance of the cardiac silhouette"])
    
    return pos_prompts, neg_prompts


def get_scores(val_data_loader, clip_model, heads, scaling = 10, return_list=True):
    clip_model.eval()
    criterion = nn.BCEWithLogitsLoss()
    pos_prompts, neg_prompts = get_prompts(heads)
    tot_val_loss = 0

    res = {}
    aucs = {}

    with torch.no_grad():
        for head in heads:
            all_targs  = []
            all_scores = []
            all_preds  = []
            all_names = []
            pos_prompt, neg_prompt = pos_prompts[head], neg_prompts[head]

            pos_text_embeddings = {}
            neg_text_embeddings = {}
            for view in ['PA', 'LATERAL']:
                pos_text_view_prompt = [f"View: {view}. {p}" for p in pos_prompt]
                neg_text_view_prompt = [f"View: {view}. {p}" for p in neg_prompt]
                
                pos_text_embeddings[view] = F.normalize(
                    clip_model.get_text_embeddings(pos_text_view_prompt, only_texts=True).mean(dim=0), dim=0, p=2)
                neg_text_embeddings[view] = F.normalize(
                    clip_model.get_text_embeddings(neg_text_view_prompt, only_texts=True).mean(dim=0), dim=0, p=2)

            # pos_text_embeddings = clip_model.get_text_embeddings(pos_prompt, only_texts=True)
            # pos_text_embeddings = pos_text_embeddings.mean(dim=0)
            # pos_text_embeddings = F.normalize(pos_text_embeddings, dim=0, p=2)
            # # print(pos_text_embeddings.shape)
            # neg_text_embeddings = clip_model.get_text_embeddings(neg_prompt, only_texts=True)
            # neg_text_embeddings = neg_text_embeddings.mean(dim=0)
            # neg_text_embeddings = F.normalize(neg_text_embeddings, dim=0, p=2)
            # print(neg_text_embeddings.shape)

            for i, samples in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
                images, labels = samples['image'], samples[head]
                views = samples['view']
                names = samples['name']
                valid_indices = (labels == 0.0) | (labels == 1.0)
                if valid_indices.sum() == 0:
                    continue  
                
                indices = valid_indices.nonzero(as_tuple=True)[0]

                # Use these indices to filter tensors and lists
                images = images[indices].to(device, dtype=torch.float32)
                labels = labels[indices].to(device)
                views = [views[idx] for idx in indices]
                names = [names[idx] for idx in indices]

                im_embeddings = clip_model.get_im_embeddings(images, only_ims=True)[0]

                scores = torch.zeros_like(labels).to(device)
                for j, view in enumerate(views):
                    pos_sim = im_embeddings[j] @ pos_text_embeddings[view].t()
                    neg_sim = im_embeddings[j] @ neg_text_embeddings[view].t()
                    scores[j] = (pos_sim - neg_sim) * scaling

                # pos_sim= im_embeddings@ pos_text_embeddings.t()
                # neg_sim= im_embeddings@ neg_text_embeddings.t()
                # print(f"image embedding {im_embeddings.shape}")
                # print(pos_sim.shape)
                # print(neg_sim.shape)
                # scores = (pos_sim - neg_sim)*scaling

                preds = (scores > 0).float()
                val_loss = criterion(scores.float(), labels.float())
                tot_val_loss += val_loss.detach().cpu().item()

                all_names.extend(names)
                all_scores.append(scores.detach().cpu().numpy())
                all_targs.append(labels.cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy()) 
            
            all_scores = np.concatenate(all_scores)
            all_targs = np.concatenate(all_targs)
            all_preds = np.concatenate(all_preds)
            auc = roc_auc_score(all_targs, all_scores)
            aucs[head] = auc
            print(f"{head} AUC is {auc}")

            assert len(all_targs) == len(all_names) == len(all_preds) == len(all_scores) 
            res[head] = {
                 'names': all_names, 
                 'scores': all_scores.tolist(),
                 'preds': all_preds.tolist(),
                 'labels': all_targs.tolist()
            }
    return res, aucs

def main(args):

    print("CUDA Available: " + str(torch.cuda.is_available()))
    heads = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
            "Fracture", "Lung Lesion",  "Pleural Effusion", "Pneumonia", "Pneumothorax", "Support Devices"]
    # heads = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    # heads = ["Atelectasis"]
    # heads = ["Cardiomegaly"]

    print("Loading CLIP model")
    clip_model = CLIP_Embedding.MedCLIP(eval=True, freeze_transformer=True, freeze_CNN=True).to(device) 
    checkpoint = torch.load(args.model_path)
    clip_model.load_state_dict(checkpoint['model_state_dict'])

    with open(os.path.join(args.data_path, f'{args.split}.json'), 'r') as file:
        test_dict = json.load(file)
    test_transform= transforms.Compose([
        transforms.Resize(256),  #256
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = MIMIC_Labels(test_dict, args.img_path, transform=test_transform)
    num_work = min(os.cpu_count(), 10)
    num_work = num_work if num_work > 1 else 0
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=num_work, prefetch_factor=2, pin_memory=True, drop_last = False)

    res, aucs = get_scores(test_data_loader, clip_model, heads)
    with open(os.path.join(args.data_path, f'res_{args.split}.json'), 'w') as file:
        json.dump(res, file, indent=4)

    # Train and validate
    # aucs = {f"{head} AUC (val)": [] for head in heads}
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # categories = {
    #         'gender': ['F', 'M'], 
    #         'race': ['AMERICAN INDIAN', 'ASIAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE'], 
    #         'insurance': ['Medicaid', 'Medicare', 'Other'],
    # }

    # all_aucs = {}
    # for category, values in categories.items():
    #     all_aucs[category] = {}
    #     for value in values:
    #         data_path = os.path.join(args.data_path, f'by_{category}', value.strip())
    #         with open(os.path.join(data_path, 'val.json'), 'r') as file:
    #             test_dict = json.load(file)

    #         test_transform= transforms.Compose([
    #             transforms.Resize(256),  #256
    #             transforms.CenterCrop(224), 
    #             transforms.ToTensor()
    #         ])
    #         test_data = MIMIC_Labels(test_dict, args.img_path, transform=test_transform)
    #         num_work = min(os.cpu_count(), 10)
    #         num_work = num_work if num_work > 1 else 0
    #         test_data_loader =  DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_work, 
    #                                 prefetch_factor=2, pin_memory=True, drop_last = False)

    #         # Train and validate
    #         # aucs = {f"{head} AUC (val)": [] for head in heads}
    #         aucs = get_similarity(test_data_loader, clip_model, heads)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('--model_path', type=str, default='/shared/beamTeam/yhuang/models/PA_LATERAL/impr/best_model.pt')
    parser.add_argument('--data_path', type=str, default='/shared/beamTeam/yhuang/data/PA_LATERAL/impr/')
    parser.add_argument('--img_path', type=str, default='/shared/beamTeam/yhuang/files/', help='directory of images')
    parser.add_argument('--split', type=str, default='val')
    # parser.add_argument('--img_path', type=str, default='/shared/beamTeam/yhuang/data/', help='directory of images')

    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)