import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def get_scores(val_data_loader, clip_model, heads, scaling = 10):
    clip_model.eval()
    criterion = nn.BCEWithLogitsLoss()
    pos_prompts, neg_prompts = get_prompts(heads)
    tot_val_loss = 0

    res = {}

    with torch.no_grad():
        for head in heads:
            all_targs  = []
            all_scores = []
            all_preds  = []
            all_names = []
            pos_prompt, neg_prompt = pos_prompts[head], neg_prompts[head]

            pos_text_embeddings = clip_model.get_text_embeddings(pos_prompt, only_texts=True)
            pos_text_embeddings = pos_text_embeddings.mean(dim=0)
            pos_text_embeddings = F.normalize(pos_text_embeddings, dim=0, p=2)
            # print(pos_text_embeddings.shape)
            neg_text_embeddings = clip_model.get_text_embeddings(neg_prompt, only_texts=True)
            neg_text_embeddings = neg_text_embeddings.mean(dim=0)
            neg_text_embeddings = F.normalize(neg_text_embeddings, dim=0, p=2)
            # print(neg_text_embeddings.shape)

            for i, samples in enumerate(val_data_loader):
                images, labels = samples['image'], samples[head]
                names = samples['name']
                print(names)
                valid_indices = (labels == 0.0) | (labels == 1.0)
                if valid_indices.sum() == 0:
                    continue  # Skip if no valid samples

                images = images[valid_indices].to(device, dtype=torch.float32)
                labels = labels[valid_indices].to(device)
                names = names[valid_indices]

                im_embeddings = clip_model.get_im_embeddings(images, only_ims=True)[0]
                # scores = torch.zeros_like(labels).to(device)

                pos_sim= im_embeddings@ pos_text_embeddings.t()
                neg_sim= im_embeddings@ neg_text_embeddings.t()
                # print(f"image embedding {im_embeddings.shape}")
                # print(pos_sim.shape)
                # print(neg_sim.shape)

                scores = (pos_sim - neg_sim)*scaling
                preds = (scores > 0)
                val_loss = criterion(scores.float(), labels.float())
                tot_val_loss += val_loss.detach().cpu().item()
                all_scores.append(scores.detach().cpu().numpy())
                all_targs.append(labels.cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy()) 
                all_names.append(names)

            assert len(all_targs) == len(all_names) == len(all_preds) == len(all_scores) 
            res[head] = {
                 'names': all_names, 
                 'scores': all_scores,
                 'preds': all_preds,
                 'labels': all_targs
            }

    return res


# def get_auc(all_scores, all_targs):
#     all_scores = np.concatenate(all_scores)
#     all_targs = np.concatenate(all_targs)
#     auc = roc_auc_score(all_targs, all_scores)
#     return auc
        
# def predictive_parity(group1_preds, group1_targs, group2_preds, group2_targs)
#         auc = roc_auc_score(all_targs, all_preds)
#         print(f"Head: {head}, Total Validation Loss: {tot_val_loss}, AUC: {auc:.4f}")
#         aucs[head] = auc