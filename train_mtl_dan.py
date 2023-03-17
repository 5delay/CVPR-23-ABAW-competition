import os
import sys
import warnings
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from utils.DAN_util import CCC_loss, CCC_score, metric_for_VA, plot_confusion_matrix
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from numpy import random
from sklearn.metrics import confusion_matrix,f1_score

from networks.MTL_dan import resnetmtl

def warn(*args, **kwargs):
    pass
warnings.warn = warn

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../Third ABAW Annotations/MTL_Challenge/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=1, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=9, help='Number of attention head.')
    return parser.parse_args()



class MTLdataset(data.Dataset):
    def __init__(self, mtl_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.mtl_path = mtl_path
        print("read df")
        df = pd.read_csv(os.path.join(self.mtl_path,phase+"_set.txt"), sep=",", engine='python')
        df.columns = ['image','valence','arousal','expression'
                        ,'au1','au2','au3','au4','au6','au7','au10','au12','au15','au23','au24','au25']
        
                                                                            

        id = df[df['expression']==-1].index
        df = df.drop(id) 
        id = df[df['au1']==-1].index
        df = df.drop(id) 
        #print(df)
        self.image, self.valence, self.arousal,self.expression,self.au = df['image'].to_numpy(), df['valence'].to_numpy(), df['arousal'].to_numpy(), df['expression'].to_numpy() ,df[['au1','au2','au3','au4','au6','au7','au10','au12','au15','au23','au24','au25']].to_numpy()
        if self.phase == "a": 
            temp        = None
            idx         = []
            image_tmp   = None
            val_tmp     = None
            aro_tmp     = None
            au_tmp      = None
            expr_tmp    = None
            
            for i, expr in enumerate(self.expression):
                if(temp is None):
                    temp=expr
                    idx.append(i)
                elif(temp==expr):
                    idx.append(i)
                else:
                    #print(len(idx))
                    if len(idx)//5!=0:
                        a = random.randint(low=len(idx)//5, size=len(idx))
                        if expr_tmp is None :
                            image_tmp   = self.image[a]
                            val_tmp     = self.valence[a]
                            aro_tmp     = self.arousal[a]
                            au_tmp      = self.au[a]
                            expr_tmp    = self.expression[a]
                        else: 
                            image_tmp = np.concatenate((image_tmp,self.image[a]),axis=0)
                            val_tmp   = np.concatenate((val_tmp, self.valence[a]),axis=0)
                            aro_tmp   = np.concatenate((aro_tmp, self.arousal[a]),axis=0)
                            au_tmp    = np.concatenate((au_tmp, self.au[a]),axis=0)
                            expr_tmp  = np.concatenate((expr_tmp,self.expression[a]),axis=0)
                    else:
                        a = idx
                        if expr_tmp is None :
                            image_tmp   = self.image[a]
                            val_tmp     = self.valence[a]
                            aro_tmp     = self.arousal[a]
                            au_tmp      = self.au[a]
                            expr_tmp    = self.expression[a]
                        else: 
                            #print(np.shape(image_tmp),np.shape(self.image[a]))
                            image_tmp = np.concatenate((image_tmp,self.image[a]),axis=0)
                            val_tmp   = np.concatenate((val_tmp, self.valence[a]),axis=0)
                            aro_tmp   = np.concatenate((aro_tmp, self.arousal[a]),axis=0)
                            au_tmp    = np.concatenate((au_tmp, self.au[a]),axis=0)
                            expr_tmp  = np.concatenate((expr_tmp,self.expression[a]),axis=0) 

                    idx = []
                    temp=None
                    
            self.image      = image_tmp
            self.valence    = val_tmp  
            self.arousal    = aro_tmp  
            self.au         = au_tmp   
            self.expression = expr_tmp 


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        path = self.image[idx]
        image = Image.open("../cropped_aligned/"+path).convert('RGB')
        valence = self.valence[idx]
        arousal = self.arousal[idx]
        expression = self.expression[idx]
        au = self.au[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, valence, arousal, expression, au

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = resnetmtl(num_head=args.num_head)
    model.to(device)
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
            print('Multi GPU activate')
            model = nn.DataParallel(model)
            model = model.cuda()
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # #transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #          transforms.RandomRotation(20),
        #          transforms.RandomCrop(224, padding=32)
        #      ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        #transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_dataset = MTLdataset(args.data_path, phase = 'train', transform = data_transforms)    
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

    val_dataset = MTLdataset(args.data_path, phase = 'validation', transform = data_transforms_val)   

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_au = torch.nn.BCELoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.55)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print("prams size:", [param_size])
    best_acc = 0
    best_expr_cm = None
    best_au_cm =None
    best_v_score = 0
    best_a_score = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        running_v_ccc = 0.0
        running_a_ccc = 0.0
        iter_cnt = 0
        model.train()
        au_p = []
        au_t = []
        tmp_V_prob, tmp_A_prob, tmp_V_label, tmp_A_label = [], [], [], []
        for (imgs, valence, arousal, expression, au) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            targets = expression.to(device)
            au = au.to(device)
            valence = valence.to(device)
            arousal = arousal.to(device)


            out, out_au, out_va, feat, head, head2   = model(imgs)
            v_CCC = CCC_loss(out_va[:,0],valence)
            a_CCC = CCC_loss(out_va[:,1],arousal)
            loss = criterion_cls(out,targets) + criterion_au(out_au,au.float()) + v_CCC +a_CCC +criterion_pt(head)  +criterion_pt(head2)

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            
            pred_au=np.where(out_au.cpu().detach()[:,:]>0.5,1,0)
            running_v_ccc +=v_CCC
            running_a_ccc +=a_CCC   
            for p, t in zip(pred_au, au) :
                au_p.append(p)
                au_t.append(t.cpu().numpy())

            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            tmp_V_prob.extend(out_va[:,0].cpu().detach().numpy())
            tmp_V_label.extend(valence.cpu().detach().numpy())
            tmp_A_prob.extend(out_va[:,1].cpu().detach().numpy())
            tmp_A_label.extend(arousal.cpu().detach().numpy())

        ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
        final_VA_score = (ccc_v + ccc_a) / 2
        au_f1 = []

        temp_exp_pred = np.array(au_p)
        temp_exp_target = np.array(au_t)
        for i in range(0,12):

            exp_pred = temp_exp_pred[:,i]
            exp_target = temp_exp_target[:,i]
            au_f1.append(f1_score(exp_pred,exp_target))
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        running_au_f1 = np.mean(au_f1)

        tqdm.write('[Epoch %d] Training accuracy: %.4f. au F1: %.4f Loss: %.3f. va CCC: %.4f LR %.7f' % (epoch, acc, running_au_f1,running_loss,final_VA_score,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
    
            y_true = []
            y_pred = []
            au_p = []
            au_t = []
            tmp_V_prob, tmp_A_prob, tmp_V_label, tmp_A_label = [], [], [], []
            model.eval()

            for (imgs, valence, arousal, expression, au) in val_loader:
                imgs = imgs.to(device)
                targets = expression.to(device)
                au = au.to(device)
                valence = valence.to(device)
                arousal = arousal.to(device)

                out, out_au, out_va, feat, head, head2   = model(imgs)
                v_CCC = CCC_loss(out_va[:,0],valence)
                a_CCC = CCC_loss(out_va[:,1],arousal)
                loss = criterion_cls(out,targets) + criterion_au(out_au,au.float()) + v_CCC +a_CCC +criterion_pt(head)  +criterion_pt(head2)


                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                pred_au=np.where(out_au.cpu().detach()[:,:]>0.5,1,0)
            
                for p, t in zip(pred_au, au) :
                    au_p.append(p)
                    au_t.append(t.cpu().numpy())
                for p, t in zip(predicts, targets) :
                    y_pred.append(p.cpu())
                    y_true.append(t.cpu())
                    
                tmp_V_prob.extend(out_va[:,0].cpu().detach().numpy())
                tmp_V_label.extend(valence.cpu().detach().numpy())
                tmp_A_prob.extend(out_va[:,1].cpu().detach().numpy())
                tmp_A_label.extend(arousal.cpu().detach().numpy())

            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2
            running_loss = running_loss/iter_cnt   
            scheduler.step()
            au_f1 = []

            temp_exp_pred = np.array(au_p)
            temp_exp_target = np.array(au_t)
            for i in range(0,12):

                exp_pred = temp_exp_pred[:,i]
                exp_target = temp_exp_target[:,i]
                au_f1.append(f1_score(exp_pred,exp_target))
            f1=[]
            temp_exp_pred = np.array(y_pred)
            temp_exp_target = np.array( y_true)
            temp_exp_pred = torch.eye(8)[temp_exp_pred]
            temp_exp_target = torch.eye(8)[temp_exp_target]
            for i in range(0,8):

                exp_pred = temp_exp_pred[:,i]
                exp_target = temp_exp_target[:,i]
                f1.append(f1_score(exp_pred,exp_target))
            running_f1 = np.mean(f1)
            running_au_f1 = np.mean(au_f1)
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            total_score = running_f1+running_au_f1+final_VA_score
            best_acc = max(total_score,best_acc)
            print(f1)
            print(au_f1)
            
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. expr F1:%.4f. au F1:%.4f. va_CCC:%.4f. total:%.4f Loss:%.3f" % (epoch, acc, running_f1,running_au_f1,final_VA_score, total_score,running_loss))
            tqdm.write("best_score:" + str(best_acc))

            if total_score == best_acc:
                best_expr_cm = confusion_matrix(y_pred,y_true)
                best_v_score = ccc_v
                best_a_score = ccc_a
                best_va = final_VA_score
                best_au = running_au_f1
                best_fer =running_f1
                best_total = total_score
                n_epoch = epoch
                model_state = model.state_dict()
                optimizer_state = optimizer.state_dict()
                
    torch.save({'iter': n_epoch,
                            'model_state_dict': model_state,
                             'optimizer_state_dict': optimizer_state,},
                            os.path.join('checkpoints',"MTL_3arr", "each_epoch"+str(n_epoch)+"_expr"+str(best_fer)+"_au"+str(best_au)+"_VA"+str(best_va)+"total %0.4f:"%best_total +".pth"))
    tqdm.write('Model saved.')
    plot_confusion_matrix(best_expr_cm, "num_head_"+str(args.num_head)+"expr_"+str(epoch)+"_cm", normalize = True, target_names = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])


if __name__ == "__main__":        
    run_training()
