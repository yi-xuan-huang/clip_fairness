import os
import time
import json
import argparse
import wandb
import torch
from torchvision import transforms
import sys
# sys.path.insert(0, '/home/apalepu/shared/beamTeam/apalepu/clip_model/clip_model/code')
# sys.path.insert(0, '/shared/beamTeam/clip_model/code/labeling_code')
import Data_Helpers
from fair.code.CXR_Datasets import NeonateXR
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    # wandb.init(project="nicu_foundation_model", 
    #     config={ 
    #         "clip_model_path": args.pretrained_loadpath,
    #         "batch_size": args.batch_size, 
    #         "data_path": args.data_path, 
    #         "start_epoch": args.start_epoch,
    #         "end_epoch": args.end_epoch,
    #     })
 
    # Device configuration
    logging.info("CUDA Available: " + str(torch.cuda.is_available()))
    heads = ["respiratory distress syndrome (rds)", "transient tachypnea of the newborn (ttn)", "atelectasis", 
             "pneumothorax", "pleural effusion", "meconium aspiration syndrome (mas)", "bronchopulmonary dysplasia (bpd)", 
             "endotracheal tube placement", "peripherally inserted central catheter (picc) placement", 
             "umbilical vein catheter (uvc)", "umbilical artery catheter (uac)","cardiomegaly", 
             "pulmonary edema","pneumonia", "pneumatosis", "pneumoperitoneum", "portal venous gas", "obstruction", 
             "atresia", "nasogastric tube placement"]

    logging.info("Loading CLIP model")
    clip_model = CLIP_Embedding.MedCLIP(eval=True, freeze_transformer=True, freeze_CNN=True).to(device) 

    #names, labels, transforms, modalities
    with open(os.path.join(args.data_path,'val.json'), 'r') as file:
        sub = json.load(file)
        instances = []
        for item in sub:
            if 'image' in item:  
                instances.append(item['image'])

    val_names, val_modalities, val_labels, val_reports = Data_Helpers.get_llm_labeled_data('val', heads=heads, keep_instances=instances)
    val_transform= transforms.Compose([
        transforms.Resize(256),  #256
        transforms.CenterCrop(224), 
        transforms.ToTensor()
    ])
    val_data = NeonateXR(args.img_path, val_names, modalities = val_modalities, labels=val_labels, texts=val_reports, transform=val_transform)
    val_data_loader=  Data_Helpers.get_loader(val_data, args=args, shuffle=False)

    # Train and validate
    # aucs = {f"{head} AUC (val)": [] for head in heads}
    aucs = {head:[] for head in heads}

    for ckpt in range(args.start_epoch, args.end_epoch+1, args.step_epoch):
        logging.info(f"Checkpoint {ckpt}")
        clip_checkpoint = torch.load(os.path.join(args.pretrained_loadpath, f'je_model-{ckpt}.pt'))
        clip_model.load_state_dict(clip_checkpoint['model_state_dict'], strict=False)
        clip_train_loss, clip_val_loss = clip_checkpoint['train_loss'], clip_checkpoint['val_loss']
        clip_model.eval()
        val_loss, val_auc, FPs, FNs, TPs, TNs, = validate_similarity_classifier(val_data_loader, clip_model, heads = heads)


        with open(os.path.join(args.pretrained_loadpath, f'FP_{ckpt}.json'), 'w') as file:
            json.dump(FPs, file,indent=4)

        with open(os.path.join(args.pretrained_loadpath, f'FN_{ckpt}.json'), 'w') as file:
            json.dump(FNs, file,indent=4)

        with open(os.path.join(args.pretrained_loadpath, f'TP_{ckpt}.json'), 'w') as file:
            json.dump(TPs, file,indent=4)

        with open(os.path.join(args.pretrained_loadpath, f'TN_{ckpt}.json'), 'w') as file:
            json.dump(TNs, file,indent=4)


        wandb.log({**val_auc}, step=ckpt + 1)
        wandb.log({'train_loss' : clip_train_loss, 'val_loss': clip_val_loss}, step=ckpt + 1)

        aucs = {k: v+[val_auc[f"{k} AUC (val)"]] for k, v in aucs.items()}
        # assert (max([len(lst) for lst in aucs.values()]) == 
        #         min([len(lst) for lst in aucs.values()]) == 
        #         ckpt//args.step_epoch+1)

    with open(os.path.join(args.pretrained_loadpath, 'label_val_aucs.json'), 'w') as file:
        json.dump(aucs, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_loadpath', type=str, default='/shared/beamTeam/clip_model/models/clip/full/exp6/', help='load weights from a clip model')
    parser.add_argument('--img_path', type=str, default='/shared/beamTeam/instances_deid/', help='directory of images')
    parser.add_argument('--data_path', type=str, default='/shared/beamTeam/clip_model/data/by_study/neonate_abd_ap_and_lat')
    parser.add_argument('--start_epoch', type=int, default=39)
    parser.add_argument('--end_epoch', type=int, default=52)
    parser.add_argument('--step_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128) 
    args = parser.parse_args()
    log(args.pretrained_loadpath)
    main(args)