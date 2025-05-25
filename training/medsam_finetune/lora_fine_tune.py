import argparse
from models.sam import SamPredictor, sam_model_registry
from models.sam import ResizeLongest
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam

# Scientific computing 
import numpy as np
import os

# Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from tensorboardX import SummaryWriter

# Visualization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.dsc import dice_coeff_multi_class
import cv2
import monai
from utils.utils import vis_image
import json
from monai.losses import TverskyLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MedDataset')
    parser.add_argument('--train_img_list', type=str, default='train.txt')
    parser.add_argument('--val_img_list', type=str, default='val.txt')
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--mask_folder', type=str, required=True)
    parser.add_argument('--targets', type=int, default=1)
    parser.add_argument('--sam_ckpt', type=str, required=True)
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--b', type=int, default=2)  # batch size
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_period', type=int, default=250)
    parser.add_argument('--if_warmup', action='store_true')
    parser.add_argument('--arch', type=str, default='vit_b')
    parser.add_argument('--finetune_type', type=str, choices=['lora', 'vanilla'], default='lora')
    parser.add_argument('--out_size', type=int, default=256)
    parser.add_argument('--num_cls', type=int, default=2)
    return parser.parse_args()


def train_model(trainloader, valloader, dir_checkpoint, epochs, args):
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr

    sam = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls)

    if args.finetune_type == 'lora':
        print('LoRA decoder only enabled.')
        args.if_update_encoder = False
        args.if_encoder_lora_layer = []  # No encoder LoRA
        args.if_decoder_lora_layer = ['all']  # Apply LoRA to all decoder layers
        sam = LoRA_Sam(args, sam, r=4).sam

    sam.to('cuda')

    optimizer = optim.AdamW(sam.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    criterion = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)

    iter_num = 0
    max_iterations = epochs * len(trainloader)
    writer = SummaryWriter(dir_checkpoint + '/log')

    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0
    for epoch in pbar:
        sam.train()
        train_loss = 0
        for i, data in enumerate(tqdm(trainloader)):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask'])
            msks = msks.cuda()

            with torch.no_grad():
                img_emb = sam.image_encoder(imgs)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            loss = criterion(pred, msks.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    lr_ = args.lr

            train_loss += loss.item()
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

        train_loss /= (i + 1)
        pbar.set_description('Epoch num {}| train loss {:.4f}'.format(epoch, train_loss))

        if epoch % 2 == 0:
            eval_loss = 0
            dsc = 0
            sam.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(valloader)):
                    imgs = data['image'].cuda()
                    msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask'])
                    msks = msks.cuda()

                    img_emb = sam.image_encoder(imgs)
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = sam.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    loss = criterion(pred, msks.float())
                    eval_loss += loss.item()
                    dsc_batch = dice_coeff_multi_class(pred.argmax(dim=1).cpu(), torch.squeeze(msks.long(), 1).cpu().long(), args.num_cls)
                    dsc += dsc_batch

                eval_loss /= (i + 1)
                dsc /= (i + 1)

                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('eval/dice', dsc, epoch)

                print('Eval Epoch {} | val loss {:.4f} | DSC {:.4f}'.format(epoch, eval_loss, dsc))
                if dsc > val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    print('Saving best model: DSC = {:.4f}'.format(dsc))
                    torch.save(sam.state_dict(), os.path.join(dir_checkpoint, 'checkpoint_best.pth'))
                elif (epoch - last_update_epoch) > 20:
                    print('Early stopping: no improvement for 20 epochs.')
                    break
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    print('Train dataset:', args.dataset_name)

    num_workers = 8
    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    with open(path_to_json, 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    print('Target class:', args.targets)

    train_dataset = Public_dataset(args, args.img_folder, args.mask_folder, args.train_img_list, phase='train', targets=[args.targets], normalize_type='sam', if_prompt=False)
    eval_dataset = Public_dataset(args, args.img_folder, args.mask_folder, args.val_img_list, phase='val', targets=[args.targets], normalize_type='sam', if_prompt=False)
    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

    train_model(trainloader, valloader, args.dir_checkpoint, args.epochs, args)
