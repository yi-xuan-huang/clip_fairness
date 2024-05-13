import os
import time
import argparse
import wandb
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# import Data_Helpers
from CXR_Datasets import MIMIC
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger(__name__)

def main(args):

    wandb.init(
        project="mimic", 
        entity = "yixuan_huang",
        config={
            "learning_rate": args.learning_rate, 
            "batch_size": args.batch_size, 
            "epochs": args.num_epochs,
            "freeze_transformer": args.freeze_transformer,
            "freeze_cnn": args.freeze_cnn,
            "exp_name": args.exp_name,
            "data_path": args.data_path})

    logger = logging.getLogger(__name__)
    logger.info(args)

    # Device configuration
    logger.info("CUDA Available: " + str(torch.cuda.is_available()))

    # Start experiment
    exp_path = getExperiment(args)
    start, je_model, params, optimizer, best_val_loss = startExperiment(args, exp_path)
    
    with open(os.path.join(args.data_path, 'train.json'), 'r') as f:
        train_dict = json.load(f)

    if args.debug: 
        train_dict = train_dict[:1000]

    train_transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, ratio=(.9, 1.0)),
        transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05)),
        transforms.ColorJitter(brightness=.2, contrast=.2),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_data = MIMIC(train_dict, args.img_path, transform=train_transform)
    logger.info(f"Train samples: {len(train_data)}")
    num_work = min(os.cpu_count(), 10)
    num_work = num_work if num_work > 1 else 0
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_work, 
                              prefetch_factor=2, pin_memory=True, drop_last = False)
    with open(os.path.join(args.data_path, 'val.json'), 'r') as f:
            val_dict = json.load(f)
    if args.debug:
        val_dict = val_dict[:1000]
    val_transform= transforms.Compose([
        transforms.Resize(256),  #256
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = MIMIC(val_dict, args.img_path, transform=val_transform)
    logger.info(f"Validation samples: {len(val_data)}")
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=num_work, 
                              prefetch_factor=2, pin_memory=True, drop_last = False)
   

    total_step= len(train_data_loader)
    assert (args.resume or start == 0)
    je_model.eval()
    # val_loss, val_losses = validate(val_data_loader, je_model, args) 

    # Train and validate
    all_train_loss, all_val_loss = [], []
    for epoch in range(start, args.num_epochs):
        je_model.train()
        ttrain = time.time()
        train_loss, train_losses = train(train_data_loader, je_model, args, epoch, optimizer, total_step)
        
        wandb.log({"train_loss": train_loss, "train_losses": train_losses})
        logger.info("Epoch time: " + str(time.time() - ttrain))

        if epoch % args.val_step == 0:
            logger.info("Validating/saving model")
            je_model.eval()
            tval = time.time()
            val_loss, val_losses = validate(val_data_loader, je_model, args)

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            wandb.log({"val_loss": val_loss, "val_losses": val_losses})

            if epoch % args.save_every == 0:
                torch.save({'epoch': epoch+1,
                            'model_state_dict': je_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'args': args}, os.path.join(exp_path, 'je_model-{}.pt'.format(epoch)))

            if val_loss <= best_val_loss:
                logger.info("Best model so far!")
                best_val_loss = val_loss
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': je_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss':best_val_loss,
                            'val_loss': val_loss,
                            'args': args}, os.path.join(exp_path, 'best_model.pt'))

                wandb.log({"new_best_model_epoch": epoch+1})

            logging.info("Val time " + str(time.time() - tval))

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('--model_path', type=str, default='/shared/beamTeam/yhuang/models/', help='path for saving trained models')
    parser.add_argument('--exp_name', type=str, default="PA_LATERAL/full_long", const=-1, nargs='?')
    parser.add_argument('--resume', type=bool, default=False, const=-1, nargs='?')
    parser.add_argument('--img_path', type=str, default='/shared/beamTeam/yhuang/files', help='directory of images')
    parser.add_argument('--data_path', type=str, default='/shared/beamTeam/yhuang/data/PA_LATERAL/full/')
    parser.add_argument('--debug', type=bool, default=False, const=True, nargs='?', help='debug mode, dont save')
    #entropy_params
    parser.add_argument('--lam_words', type=float, default=0)
    parser.add_argument('--lam_patches', type=float, default=0)

    #Training parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.00005) #.0001
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=1, help='step size for printing val info')
    parser.add_argument('--save_every', type=int, default=5, help='save model after number of epochs')
    parser.add_argument('--freeze_transformer', type=bool, default=False)
    parser.add_argument('--freeze_cnn', type=bool, default=False)
    args = parser.parse_args()
    log(os.path.join(args.model_path, args.exp_name))
    main(args)