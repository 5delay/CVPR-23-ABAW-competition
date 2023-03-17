# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import argparse
from tqdm import tqdm
import glob
import csv
import numpy as np

from networks.Sequentials import LSTM, Bi_LSTM, LSTM_drop, LSTM_fc, Conv_LSTM
from utils.dataloader import test_feature_Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/abaw5/MTL_features/', help='ABAW5 dataset path.')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')

    parser.add_argument('--RNN_num_layers', type=int, default=2, help='RNN num layers')
    parser.add_argument('--RNN_hidden_size', type=int, default=128, help='RNN hidden size')
    
    parser.add_argument('--task', type=str, default=None, help="multiple models ?")
    
    parser.add_argument('--ensemble_model', type=str, default='LSTM_2_7_PCC_CosineAnnealingLR')
    
    return parser.parse_args()



def get_mean_result(models_predict):
    transpose_p = np.transpose(models_predict, (1, 2, 0))
    result = np.mean(transpose_p, axis=2).tolist()
    return result


def save_result_csv(ensemble, video_names, output, target):

    head = ['File_ID', 'Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
    
    save_path = './Test_Results/Ensemble/'

    predict_save_path = save_path + f'/{ensemble}_predicted.csv'
    
    with open(predict_save_path, 'w',newline='') as f: 
        wr = csv.writer(f) 
        results = [[video_names[i].split('/')[-1]]+output[i] for i in range(len(video_names))]
        
        wr.writerow(head)
        wr.writerows(results)


def Load_folds(folds):
    
    # Set Models
    input_size = 22
    sequence_length = 12
    hidden_size = args.RNN_hidden_size
    num_layers = args.RNN_num_layers
    num_classes = 7
    
    
    models = []

    for fold in folds:
        weight = glob.glob(fold + "/best*.pt")[0]
        
        RNN_model_name = fold.split('/')[-1].split(f'_{num_layers}_')[0].split('_')[-1]

        if RNN_model_name == 'LSTM': 
            model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        elif RNN_model_name == 'Bi_LSTM': 
            model = Bi_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        elif RNN_model_name == 'LSTM_drop': 
            model = LSTM_drop(input_size, hidden_size, num_layers, num_classes).to(device)
        elif RNN_model_name == 'LSTM_fc': 
            model = LSTM_fc(input_size, hidden_size, num_layers, num_classes).to(device)
        elif RNN_model_name == 'Conv_LSTM': 
            model = Conv_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        else: raise Exception("Please Check the RNN Model Name")

        ckpt = torch.load(weight)

        new_state_dict = {}
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

        # Using Multi-GPU
        if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
            model = nn.DataParallel(model)
            model = model.cuda()
        models.append(model)

    return models



def run_ensemble():
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    # Data Loader    
    test_dataset = test_feature_Dataset(args.data_path)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
                            num_workers = args.workers, shuffle = False,  
                            pin_memory = True, drop_last = True)


    test_video_name = []
    test_predict = []
    test_target = []  
    
    test_pcc = 0.0

#     for f in folds:
#         if f.split('/')[-1][0]=='s': folds.remove(f)

    RNN_models = Load_folds(folds)
    
    
    for samples in tqdm(test_loader):
            
        video_name, x_test = samples
            
        x_test = x_test.to(device)
        
        all_model_predict = []

        for RNN in RNN_models:
            RNN.eval()
            rnn_output = RNN(x_test)   
            
            all_model_predict.append(rnn_output.cpu().tolist())

        ensemble_output = get_mean_result(all_model_predict)

        test_predict += ensemble_output
        test_video_name += video_name

        
    save_result_csv(args.ensemble_model, test_video_name, test_predict)
    
    print(f'{args.ensemble_model} Test Ended')


if __name__=="__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = '/abaw5/MTL_abaw5/test_checkpoints/'
    
    if args.task is not None:
        folds = []
        for ensemble in args.ensemble_model.split(","):
            folds += sorted(glob.glob(ckpt_dir + f'/*{ensemble}'))
            
    else:
        folds = sorted(glob.glob(ckpt_dir + f'/*{args.ensemble_model}'))
    os.makedirs('./Test_Results/Ensemble/', exist_ok=True)
    
    run_ensemble()



