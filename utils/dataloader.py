# -*- coding: utf-8 -*-
# +
import os
from tqdm import tqdm

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import csv


class abaw_Dataset(Dataset):
    def __init__(self, path, train=True, transform = None):
        super(abaw_Dataset, self).__init__()
        if train: self.path = os.path.join(path, 'train')
        else: self.path = os.path.join(path,'val')
        
        self.transform = transform
        self.csv_df = self._read_csv()
        
        self.X = []
        self.y = []
        
        for video in tqdm(os.listdir(self.path), desc = 'Loading '+ self.path):
            x = os.path.join(self.path, video)
            
            if len(os.listdir(x)) == 0 : continue
            else:
                self.X.append(x)
                label = self.csv_df[self.csv_df['name']==video]['emotion'].values[:][0]
                self.y.append(label)
    
    
    def _read_csv(self):
        data_dict = {}

        with open('/abaw5/Data/data_info.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)  

            for row in reader:
                if row['Adoration']=='': continue

                emotion = [row['Adoration'], row['Amusement'], row['Anxiety'], row['Disgust'], row['Empathic-Pain'], row['Fear'], row['Surprise']]
                emotions = list(map(float, emotion))

                data_dict[row['File_ID'][1:-1]]=[emotions,row['Split'],row['Country']]
    
        data = []
        for video in os.listdir(self.path):
            name = video
            label = data_dict[name][0]
            split = data_dict[name][1]
            country = data_dict[name][2]

            try:
                data.append({
                    "name": name,
                    "emotion": label,
                    "split": split,
                    "country": country,
                })
            except Exception as e:
                print(e)
                pass
        
        df = pd.DataFrame(data).sort_values(by=["name"])
        return df   
    
    
    def __getitem__(self, idx):
        
        x = self.X[idx]
        sorted_list = sorted(os.listdir(x))
 
        X_images = []

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
        y = torch.Tensor(self.y[idx])
        
        # return video_name, normalized tensor images, label
        return x, X, y

    def __len__(self):
        return len(self.X)


class fold_feature_Dataset(Dataset):
    def __init__(self, path, fold=None, train=None):
        super(fold_feature_Dataset, self).__init__()
        self.fold = fold
        self.train = train
        
        if self.fold is not None:
            self.path = os.path.join(path, 'fold'+str(self.fold))
        
        elif self.train is not None:
            self.path = os.path.join(path, 'train')

            
        else: self.path = os.path.join(path, 'val')
        
        self.csv_df = self._read_csv()
        
        self.X = []
        self.video_names = []
        self.y = []

        for video_np in tqdm(os.listdir(self.path), desc = 'Loading '+ self.path):
            x = os.path.join(self.path, video_np)
            x_np = np.load(x) 
            
            self.X.append(x_np)
            self.video_names.append(video_np.split('.')[0])
            
            label = self.csv_df[self.csv_df['name']==video_np.split('.')[0]]['emotion'].values[:][0]
            self.y.append(label)
    
    
    def _read_csv(self):
        data_dict = {}

        with open('/abaw5/Data/data_info.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)  

            for row in reader:
                if row['Adoration']=='': continue

                emotion = [row['Adoration'], row['Amusement'], row['Anxiety'], row['Disgust'], row['Empathic-Pain'], row['Fear'], row['Surprise']]
                emotions = list(map(float, emotion))

                data_dict[row['File_ID'][1:-1]]=[emotions,row['Split'],row['Country']]
    
        data = []
        

        for video in os.listdir(self.path):
            name = video.split(".")[0]
            label = data_dict[name][0]
            split = data_dict[name][1]
            country = data_dict[name][2]

            try:
                data.append({
                    "name": name,
                    "emotion": label,
                    "split": split,
                    "country": country,
                })
            except Exception as e:
                print(e)
                pass
        
        df = pd.DataFrame(data).sort_values(by=["name"])
        return df
    
    
    def __getitem__(self, idx):
        
        x = self.video_names[idx]
        X = self.X[idx]
        y = torch.Tensor(self.y[idx])
        
        return x, X, y

    def __len__(self):
        return len(self.X)


class test_feature_Dataset(Dataset):
    def __init__(self, path, fold=None, train=None):
        super(fold_feature_Dataset, self).__init__()
        self.fold = fold
        self.train = train
        
        self.path = os.path.join(path, 'test')
        
        self.X = []
        self.video_names = []

        for video_np in tqdm(os.listdir(self.path), desc = 'Loading '+ self.path):
            x = os.path.join(self.path, video_np)
            x_np = np.load(x) 
            
            self.X.append(x_np)
            self.video_names.append(video_np.split('.')[0])
        
    
    def __getitem__(self, idx):
        
        x = self.video_names[idx]
        X = self.X[idx]
        
        return x, X

    def __len__(self):
        return len(self.X)
