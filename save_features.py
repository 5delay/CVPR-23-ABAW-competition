# -*- coding: utf-8 -*-
#from ast import expr
import os
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.MTL_dan_for_RNN import resnetmtl_for_rnn


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/abaw5/Crop_data/', help='ABAW5 dataset path.')
    parser.add_argument('--DAN_ckpt', type=str, default='./DAN_bestmodel.pth', help='DAN model path.')
    parser.add_argument('--DAN_num_head', type=int, default=9, help='Number of attention head in MTL DAN.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--np_save_path', type=str, default='/abaw5/MTL_features/')
    return parser.parse_args()


class Datas(Dataset):
    def __init__(self, path, train=True, transform = None):
        super(Datas, self).__init__()
        
        if train: self.path = os.path.join(path, 'train')
        else: self.path = os.path.join(path,'test')
        
        self.transform = transform
        
        self.X = []
        
        for video in tqdm(os.listdir(self.path), desc = 'Loading '+ self.path):
            x = os.path.join(self.path, video)
            
            if len(os.listdir(x)) == 0 : continue
            else:
                self.X.append(x)
    
    
    
    def __getitem__(self, idx):
        
        x = self.X[idx]
        sorted_list = sorted(os.listdir(x))
 
        X_images = []

        #frame
        # Average = 11.8 , MAX = 17
        if len(sorted_list) > 12:
            del_num = len(sorted_list) - 12
            front = del_num - (del_num//2)
            back = -(del_num//2)

            if front != -1: del sorted_list[:front]
            if back != 0: del sorted_list[back:]

        if len(sorted_list) < 12:
            while len(sorted_list) < 12:
                sorted_list.append(sorted_list[-1])

        for img_path in sorted_list:
            image = Image.open(os.path.join(x, img_path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            X_images.append(image)
            
        X = X_images

        return x, X

    def __len__(self):
        return len(self.X)


def get_features():

    # Data Loader
    train_dataset = Datas(args.data_path, train = True, transform = data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                              num_workers = args.workers, shuffle = False,  
                              pin_memory = True, drop_last = False)

    
    
    MTL_Dan = resnetmtl_for_rnn(num_head=args.DAN_num_head).to(device)
    
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        MTL_Dan = nn.DataParallel(MTL_Dan)
        MTL_Dan = MTL_Dan.cuda()
    
    # Load Multitask DAN ckpt
    checkpoint = torch.load(args.DAN_ckpt)
    
    if isinstance(MTL_Dan, nn.DataParallel): # Multi GPU
        MTL_Dan.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    else: # Single GPU
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict() 
        for k, v in state_dict.items(): 
            name = k[7:]
            new_state_dict[name] = v 
        MTL_Dan.load_state_dict(new_state_dict, strict = True)
    
    MTL_Dan.eval()

    for samples in tqdm(train_loader):
        
        video_names, images = samples
        
        mtl_output = MTL_Dan(images)
        mtl_output = mtl_output.cpu()
        
        for idx in range(len(video_names)):
            video = video_names[idx].split("/")[-1]
            np_output = mtl_output[idx].detach().numpy()
            np.save(f'{args.np_save_path}/train/{video}.npy', np_output)
        

if __name__=="__main__":
    
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(f'{args.np_save_path}/train/', exist_ok=True)

    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    get_features()


