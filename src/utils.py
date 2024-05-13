import logging
import datetime
import regex as re
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import Vision_Model
import CLIP_Embedding
from CXR_Datasets import *

logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(path, fn=None):
    '''
    config logging
    '''
    # if not os.path.exists(path):
    #     os.mkdir(path)
    os.makedirs(path, exist_ok = True)
    now = datetime.datetime.now()
    if not fn:
        fn = now.strftime("out_%Y_%m_%d_%H%M.log")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(path, fn), mode='a'),
                            logging.StreamHandler()
                        ])

    # also print to console
    logging.StreamHandler().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
#Organizing training experiments
#Retrieves from exp path, or starts a new one
def getExperiment(args):
    fp = os.path.join(args.model_path, args.exp_name)

    if args.resume:
        if os.path.exists(fp):
            return fp
        else:
            raise Exception("Experiment doesn't exist, cannot resume")

    if len(args.exp_name) > 0:
        return fp

    else:
        all_files = os.listdir(args.model_path)
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        fp = os.path.join(args.model_path, f'exp{highest}')
    return fp

#Document the args used in experiment
def writeArgs(fp, args):
    '''
    Document args used to train
    '''
    writestr = str(args)
    with open(fp + '/args.txt', 'w') as f:
        f.write(writestr)

#Initializes from experiment
def startExperiment(args, fp, cnn=False, pretrained=True):
    '''
    Initialize variables for experiment:
    start (epoch), je_model, params, optimizer, best_val_loss
    '''
    if cnn:
        je_model = Vision_Model.getCNN(pretrained=pretrained, classifier=True).to(device)
    else:
        je_model = CLIP_Embedding.MedCLIP(eval=False, freeze_transformer=args.freeze_transformer, freeze_CNN=args.freeze_cnn).to(device) 

    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    if fp == "debug":
        return 0, je_model, params, optimizer, 100000

    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'je_model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            logging.info("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint.keys() else checkpoint['val_loss']
        else:
            raise Exception("Experiment doesn't exist: " + fp)
    else:
        logging.info("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            try: 
                os.makedirs(fp)
            except FileExistsError:
                pass
            writeArgs(fp, args)
    return start, je_model, params, optimizer, best_val_loss

# TIER Penalty
def attn_penalty(cross_weights, soft = nn.Softmax(dim=2), lam = (0.0, 0.0)):
    attn_loss = 0 #lam patch * entropy of each word similarity (across patches)
    losses = []   #lam words * entropy of each patch similarity (across words)
    eps = 1e-7
    for c in cross_weights:
        entropy = soft(c) + eps #NTP
        entropy = -entropy * torch.log(entropy)
        entropy_text = torch.sum(entropy, dim=2) #N, T
        entropy_text = torch.mean(entropy_text, dim=(1, 0)) #1

        entropy_im = soft(c.permute(0, 2, 1)) + eps
        entropy_im = -entropy_im * torch.log(entropy_im)
        entropy_im = torch.sum(entropy_im, dim=2) #N, P
        entropy_im = torch.mean(entropy_im, dim=(1, 0)) # 1

        loss = (entropy_text * float(lam[0])) + (entropy_im * float(lam[1]))
        losses.append(loss.cpu().detach())
        attn_loss += loss
    return attn_loss, losses #1

# Standard CLIP loss, for a list of images and texts.
# When len is 1, just normal CLIP loss. With multiple image augmentations, also do image-image contrasting
def clip_loss(im_logits, aug_logits = None, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    text_logits = [im.t() for im in im_logits]
    clip_loss = 0
    losses = []
    for i in np.arange(len(im_logits)): #for each image augmentation-text matrix (usually just 1)
        samp = torch.tensor(np.arange(im_logits[i].shape[0]))
        loss_a = criterion(im_logits[i], samp.to(device))
        loss_b = criterion(text_logits[i], samp.to(device))
        closs = (loss_a + loss_b) / 2
        losses.append(closs.cpu().detach())
        clip_loss += closs * loss_weight
    if aug_logits is not None: #for each image augmentation-image augmentation matrix (usually 0)
        for i in np.arange(len(aug_logits)):
            samp = torch.tensor(np.arange(im_logits[i].shape[0]))
            imloss = criterion(im_logits[i], samp.to(device))
            losses.append(imloss.cpu().detach())
            clip_loss += imloss
    assert len(losses) == int((len(im_logits) + (len(im_logits) * (len(im_logits) -1)/2.0)))
    return clip_loss, losses

#Compute total loss
def compute_loss(je_model, samples, attn_lam_words = 0.0, attn_lam_patches = 0.0):
    ims = samples['image']
    texts = samples['text']

    im_logits, crosses, aug_logits = je_model(ims, texts)

    cl, cl_losses = clip_loss(im_logits, aug_logits)
    attn_pen, attn_losses = attn_penalty(crosses, lam=(attn_lam_words, attn_lam_patches))
    cl_count = len(cl_losses)
    attn_count = len(attn_losses)
    loss = cl / cl_count + attn_pen / attn_count
    all_losses = cl_losses + attn_losses
    return loss, torch.tensor(all_losses)

def get_similarity(val_data_loader, clip_model, heads):

    clip_model.eval()
    criterion = nn.BCEWithLogitsLoss()

    
    label_names = None
    pos_prompts, neg_prompts = get_prompts(heads)
    aucs = {}
    total_val_loss = 0

    with torch.no_grad():
        for head in heads:
            all_targs = []
            all_preds = []
            pos_prompts, neg_prompts = get_prompts(head)

            pos_text_embeddings = clip_model.get_text_embeddings(pos_prompts, only_texts=True)
            # pos_text_embeddings = pos_text_embeddings.mean(dim=0)
            # pos_text_embeddings = F.normalize(pos_text_embeddings, dim=0, p=2)
            neg_text_embeddings = clip_model.get_text_embeddings(neg_prompts, only_texts=True)
            # neg_text_embeddings = neg_text_embeddings.mean(dim=0)
            # neg_text_embeddings = F.normalize(neg_text_embeddings, dim=0, p=2)

            for i, samples in enumerate(val_data_loader):
                images, labels = samples['image'], samples[head]
                valid_indices = (labels == 0.0) | (labels == 1.0)
                if valid_indices.sum() == 0:
                    continue  # Skip if no valid samples

                images = images[valid_indices].to(device, dtype=torch.float32)
                labels = labels[valid_indices].to(device)

                im_embeddings = clip_model.get_im_embeddings(images, only_ims=True)[0]
                # scores = torch.zeros_like(labels).to(device)

                pos_sim= im_embeddings@ pos_text_embeddings.t()
                neg_sim= im_embeddings@ neg_text_embeddings.t()

                scores = torch.max(pos_sim, neg_sim)
                val_loss = criterion(scores.float(), labels.float())
                tot_val_loss += val_loss.detach().cpu().item()
                all_preds.append(scores.detach().cpu().numpy())
                all_targs.append(labels.cpu().numpy())

                val_loss = criterion(scores.float(), labels.float())
                tot_val_loss += val_loss.detach().cpu().item()
                all_preds.append(scores.detach().cpu().numpy())
                all_targs.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_targs = np.concatenate(all_targs)
            auc = roc_auc_score(all_targs, all_preds)
            print(f"Head: {head}, Total Validation Loss: {tot_val_loss}, AUC: {auc:.4f}")
            aucs[head] = auc
        # all_preds = np.concatenate(all_preds)
        # all_targs = np.concatenate(all_targs)
        # aucs = {}
        # weighted = {}
        # total_pos = 0
        # for i, lab in enumerate(heads):
        #     try:
        #         lab_auc = roc_auc_score(all_targs[:, i], all_preds[:, i])
        #     except ValueError:
        #         lab_auc = .5
        #     aucs[lab + ' AUC (val)'] = lab_auc
        #     total_pos += np.sum(all_targs[:, i])
        #     weighted[lab + ' AUC (val)'] = lab_auc * np.sum(all_targs[:, i])
        #     print(lab, lab_auc)
        # macro_auc = np.mean(np.array(list(aucs.values())))
        # micro_auc = np.sum(np.array(list(weighted.values())))/total_pos
        # aucs['macro AUC (val)'] = macro_auc
        # aucs['micro AUC (val)'] = micro_auc
        # print('macro AUC val', macro_auc)
        # print('micro AUC val', micro_auc)
        # avg_val_loss = tot_val_loss/len(val_data_loader)
    return aucs

#Train TIER CLIP
def train(train_data_loader, je_model, args, epoch, optimizer, total_step=-1):
    mean_loss, mean_losses, ct = 0.0, 0.0, 0
    je_model.train(True)

    for i, samples in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
        je_model.zero_grad(set_to_none=True)
        loss, all_losses = compute_loss(je_model, samples, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
        # Forward, backward and optimize
        loss.backward()
        optimizer.step()
        if total_step> 0:
            if i % args.log_step == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item()))
                logging.info(all_losses)
        l = loss.cpu().detach().numpy()
        mean_loss += l
        mean_losses += all_losses
        ct += 1
    if ct > 0:
        mean_loss = mean_loss/ct
        mean_losses = mean_losses/ct

    return mean_loss, mean_losses


#Validate TIER CLIP
def validate(val_data_loader, je_model, args):
    val_losses = []
    avg_loss, ct = 0.0, 0
    je_model.train(False)
    with torch.no_grad():
        for samples in tqdm(val_data_loader, total=len(val_data_loader)):
            loss, all_losses = compute_loss(je_model, samples, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
            val_losses.append(all_losses.view(-1,1))
            avg_loss += loss
            ct += 1
    avg_loss = avg_loss/ct

    val_losses = torch.cat(val_losses, dim=1) #num batches x num losses
    avg_losses = torch.mean(val_losses, dim=1)

    if avg_losses.shape[0] == 5:
        names = ['im1-t', 'im2-t', 'im1-im2', 'im1-cross', 'im2-cross']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        logging.info("Val losses" + lossstr)
    elif avg_losses.shape[0] == 2:
        names = ['im-t', 'attn im-t']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        logging.info("Val losses" + lossstr)

    return avg_loss.item(), avg_losses

# Loss for supervised baseline
def b_loss(cnn_model, samples, heads, criterion=torch.nn.BCEWithLogitsLoss(reduction='mean')):
    im = samples['images'][0].to(device)
    impreds = cnn_model(im).class_logits.to(device)
    impreds = impreds.squeeze(dim=2)
    labels = samples['labels']
    losses = torch.zeros(len(heads))
    for i, h in enumerate(heads):
        label = labels[h]
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        label = label.float().to(device)
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

# Train supervised
# def train_vision(train_data_loader, cnn_model, args, epoch, optimizer, totstep= None, heads=None, list_mods = False, je_inds = [], all_pos_embeds=None, all_neg_embeds=None):
#     if list_mods:
#         full_loss = [0 for i in cnn_model]
#     else:
#         full_loss = [0]
#         cnn_model = [cnn_model]
#         optimizer = [optimizer]

#     for i, samples in enumerate(train_data_loader):
#         for j, cnn_mod in enumerate(cnn_model):
#             cnn_mod.train(True)
#             cnn_mod.zero_grad(set_to_none=True)
#             if j not in je_inds:
#                 loss = b_loss(cnn_mod, samples, args, heads)
#             else:
#                 cnn_mod.cnn.train(True)
#                 loss = pseudo_b_loss(cnn_mod, samples, args, heads, all_pos_embeds[j], all_neg_embeds[j])
#             loss.backward()
#             # Forward, backward and optimize

#             optimizer[j].step()
#             if (totstep) and (i % args.log_step == 0):
#                 logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, totstep,
#                                                                                loss.item()))
#             full_loss[j] += loss.detach().cpu().item()
#     avg_loss = [f/(i + 1) for f in full_loss]
#     return avg_loss[0] if not list_mods else avg_loss

def train_vision(train_data_loader, vision_model, args, epoch, optimizer):
    '''
    vision model loss for one epoch
    Return: average loss for one epoch
    '''
    criterion = nn.CrossEntropyLoss()
    tot_step = len(train_data_loader)
    tot_loss = 0.0
    for i, samples in enumerate(tqdm(train_data_loader)):
        vision_model.train(True)
        vision_model.zero_grad(set_to_none=True)

        images, labels = samples['image'], samples['label']

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)
        # Forward, backward and optimize
        outputs = vision_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().cpu().item()
        if (tot_step) and (i % args.log_step == 0):
            logging.info(f"Epoch {epoch}/{args.num_epochs}, Step {i}/{tot_step}, Loss {round(loss.item(), 5)}")

    return tot_loss/len(train_data_loader)

# Validate supervised
# def validate_vision(val_data_loader, cnn_model, args=None, heads=None, list_mods = False):
#     if list_mods:
#         avg_loss = [0.0 for mod in cnn_model]
#     else:
#         avg_loss = [0.0]
#         cnn_model = [cnn_model]

#     ct = 0
#     with torch.no_grad():
#         for i, samples in enumerate(val_data_loader):
#             for j, mod in enumerate(cnn_model):
#                 mod.train(False)
#                 loss = b_loss(mod, samples, args, heads)
#                 avg_loss[j] += loss
#             ct += 1
#     avg_loss = [a/ ct for a in avg_loss]
#     logging.info("Val loss: " + str(avg_loss))

def validate_vision(val_data_loader, vision_model, args, return_AUC=False):
    vision_model.eval()
    criterion = nn.CrossEntropyLoss()

    all_targs = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        tot_val_loss = 0.0
        for i, samples in enumerate(val_data_loader):
            images, labels = samples['image'], samples['label']
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = vision_model(images)
            val_loss = criterion(outputs, labels)
            tot_val_loss += val_loss.detach().cpu().item()

            probs, preds = vision_model.predict(images)
            targs=labels

            all_probs.extend(probs.cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy())
            all_targs.extend(targs.cpu().numpy())

    all_probs = np.array(all_probs)
    auc_macro = roc_auc_score(all_targs, all_probs, multi_class="ovr", labels=list(range(args.num_classes)), average='macro')
    auc_weighted = roc_auc_score(all_targs, all_probs, multi_class="ovr", labels=list(range(args.num_classes)), average='weighted')
    avg_val_loss = tot_val_loss / len(val_data_loader)
    logging.info(f"Val loss: {np.round(avg_val_loss, 2)}")
    logging.info(f"Val macro AUC: {np.round(auc_macro, 2)}")
    logging.info(f"Val weighted AUC: {np.round(auc_weighted, 2)}")

    if return_AUC:
        return avg_val_loss, auc_macro, auc_weighted

    return avg_val_loss



# Get labels from datafrae
def getLabels(df, heads, replace_nan = False):
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label==-1.0] = float('nan')
        if replace_nan:
            label = torch.nan_to_num(label)
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c

# Get labels from chexperrt radiologist
def getRadLabels(df, heads, suffix):
    labels = None
    for i, h in enumerate(heads):
        label = df[h + '_' + suffix].float()
        label[label==-1.0] = float('nan')
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c



def getPadPredictions(DL, models = None, ensemble=True, soft_norm=True, only_labels = False):
    tt = []
    for z, sample in enumerate(DL):
        if z == 0:
            name_list = list(sample['labels'].keys())
            label_embeds_all = []
            neg_label_embeds_all = []
            if not only_labels:
                tps_models = [[] for model in models]
                with torch.no_grad():
                    for j, m in enumerate(models):
                        label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list)
                        embed_list = [label_embeds[h][None, :] for h in name_list]
                        label_embeds = torch.cat(embed_list, dim=0)
                        label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)
                        label_embeds_all.append(label_embeds)
                        logging.info("labelembed shape",label_embeds_all[0].shape)

                        neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list, getneg=True)
                        neg_embed_list = [neg_label_embeds[h][None, :] for h in name_list]
                        neg_label_embeds = torch.cat(neg_embed_list, dim=0)
                        neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)
                        neg_label_embeds_all.append(neg_label_embeds)
                    l = label_embeds[2, :]
                    n = neg_label_embeds[2, :]
                    l1 = label_embeds[1, :]
                    logging.info(torch.sum(l * n), torch.sum(l * l1))

        labels = getLabels(sample['labels'], name_list)
        tt.append(labels)
        if only_labels:
            continue

        images = sample['images']
        for j, m in enumerate(models):
            with torch.no_grad():
                im_embeds = m.get_im_embeddings(images, only_ims=True)[0]
                im_embeds = im_embeds / im_embeds.norm(dim=1, keepdim=True)
                # N P E x c E = N c
                preds = im_embeds @ label_embeds_all[j].t()
                if soft_norm:
                    neg_preds = im_embeds @ neg_label_embeds_all[j].t() # N c
                    preds= torch.stack([preds[:, :, None], neg_preds[:, :, None]],
                                                    dim=2)  # N C 2
                    preds = torch.nn.Softmax(dim=2)(preds)[:, :, 0].squeeze(dim=2)

                tps_models[j].append(preds.cpu())

    if not only_labels:
        tps_models = [torch.cat(tps, dim=0) for tps in tps_models]#list of models, list of ims,tensor preds
    tt = torch.cat(tt, dim=0)
    if only_labels:
        return tt

    if ensemble:
        if len(models)>1:
            tps_models = [modelpred[None, :, :] for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = torch.cat(tps_models, dim = 0).mean(dim = 0, keepdim=False) #stacked prediction ten
        else:
            tps_models = [modelpred for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = tps_models[0]
        logging.info(tps_avg.shape)
        logging.info(tt.shape)
        assert tt.shape == tps_avg.shape
        tps_models = tps_avg

    return tps_models, tt, name_list

def normalize(image, getOne = True):
    img = torch.clone(image)
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    if getOne:
        img = img.permute(0, 2, 3, 1)[0, :, :, :].squeeze()
    else:
        img = img.permute(0, 2, 3, 1)
    return img

def getLabelSimilarities(mod, heads, label_embeds=None, compare_mimic = False):
    with torch.no_grad():
        if compare_mimic:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            label_embeds_mimic = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=False)
            for i, h in enumerate(heads):
                logging.info(h, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                       label_embeds_mimic[h] / label_embeds_mimic[h].norm(dim=-1, keepdim=True)).cpu())
        else:
            if not label_embeds:
                label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            for i, h in enumerate(heads):
                for j, h2 in enumerate(heads):
                    if i < j:
                        logging.info(h, h2, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                               label_embeds[h2] / label_embeds[h2].norm(dim=-1, keepdim=True)).cpu())

def get_clip_models(checkpoints=[], eval=True, freeze_text=False, freeze_CNN = False):
    '''
    returns a list of MedCLIP checkpoints, if specified,
    otherwise returns a new MedCLIP model
    '''
    if not checkpoints:
        return CLIP_Embedding.MedCLIP(eval=eval).to(device)

    models = []
    for checkpoint in checkpoints:
        model = CLIP_Embedding.MedCLIP(eval=eval).to(device)
        model.load_state_dict(torch.load(checkpoint), strict=False)
        model.freezeTransformer = freeze_text
        model.cnn.freeze_encoder = freeze_CNN

        models.append(model)

    return models 

def get_prompts(heads):
    print(heads)

    pos_prompts = {
    'Atelectasis': ["Atelectasis is present.", "Basilar opacity and volume loss is likely due to atelectasis."],
    'Cardiomegaly': ["Cardiomegaly is present.", "The heart shadow is enlarged.", "The cardiac silhouette is enlarged."], 
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

    return pos_prompts, neg_prompts