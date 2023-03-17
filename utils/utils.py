import torch
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class PCCLoss(nn.Module):
    def __init__(self):
        super(PCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)

        var_pred = torch.var(y_pred)
        var_true = torch.var(y_true)

        cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

        pcc = cov / (torch.sqrt(var_pred) * torch.sqrt(var_true))
        
        loss = 1 - pcc

        return loss


class Single_PCCLoss(nn.Module):
    def __init__(self):
        super(Single_PCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        losses = []
        for i in range(7):
            y_pred_i = y_pred[:, i]
            y_true_i = y_true[:, i]
            
            mean_pred = torch.mean(y_pred_i)
            mean_true = torch.mean(y_true_i)

            var_pred = torch.var(y_pred_i)
            var_true = torch.var(y_true_i)

            cov = torch.mean((y_pred_i - mean_pred) * (y_true_i - mean_true))

            pcc = cov / (torch.sqrt(var_pred) * torch.sqrt(var_true))
        
            loss = 1 - pcc
            losses.append(loss)

        return sum(losses)/7


class Total_PCCLoss(nn.Module):
    def __init__(self):
        super(Total_PCCLoss, self).__init__()
        self.pcc_loss = PCCLoss()
        self.single_pcc_loss = Single_PCCLoss()
        self.num_emotions = 7

    def forward(self, y_pred, y_true):
        total_pcc_loss = self.pcc_loss(y_pred[:, :self.num_emotions], y_true[:, :self.num_emotions])
        single_pcc_loss = self.single_pcc_loss(y_pred[:, :self.num_emotions], y_true[:, :self.num_emotions])
        total_loss = (total_pcc_loss + single_pcc_loss) / 2
        return total_loss


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)

        var_pred = torch.var(y_pred)
        var_true = torch.var(y_true)

        cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

        ccc = 2 * cov / (var_pred + var_true + (mean_pred - mean_true)**2)

        loss = 1 - ccc

        return loss

class Single_CCCLoss(nn.Module):
    def __init__(self):
        super(Single_CCCLoss, self).__init__()


    def forward(self, y_pred, y_true):
        losses = []
        for i in range(7):
            y_pred_i = y_pred[:, i]
            y_true_i = y_true[:, i]
            mean_pred = torch.mean(y_pred_i)
            mean_true = torch.mean(y_true_i)
            var_pred = torch.var(y_pred_i)
            var_true = torch.var(y_true_i)
            cov = torch.mean((y_pred_i - mean_pred) * (y_true_i - mean_true))
            ccc = 2 * cov / (var_pred + var_true + (mean_pred - mean_true)**2)
            loss = 1 - ccc
            losses.append(loss)
        return sum(losses) / 7

class Total_CCCLoss(nn.Module):
    def __init__(self):
        super(Total_CCCLoss, self).__init__()
        self.ccc_loss = CCCLoss()
        self.single_ccc_loss = Single_CCCLoss()
        self.num_emotions = 7

    def forward(self, y_pred, y_true):
        total_ccc_loss = self.ccc_loss(y_pred[:, :self.num_emotions], y_true[:, :self.num_emotions])
        single_ccc_loss = self.single_ccc_loss(y_pred[:, :self.num_emotions], y_true[:, :self.num_emotions])
        total_loss = (total_ccc_loss + single_ccc_loss) / 2
        return total_loss


def PCC_metric(video_name, output, target):
    from torchmetrics.functional import pearson_corrcoef
    pearson_7 = pearson_corrcoef(output, target)
    
    return torch.mean(pearson_7).item()


def save_loss_pcc_plt(mode_name, train_losses, train_pccs, val_losses, val_pccs):
    
    save_path = './fold_Results/' + mode_name
    
    epochs = range(1, len(train_losses) + 1)

    # Plot the training and validation loss curves
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')

    # Set the plot title, axis labels, and legend
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0.4, 1.0])
    plt.legend()

    # Show the plot
    plt.savefig(save_path+'/loss_plt.png')
    
    plt.clf()
    
     # Plot the training and validation loss curves
    plt.plot(epochs, train_pccs, 'b', label='Training PCC')
    plt.plot(epochs, val_pccs, 'r', label='Validation PCC')

    # Set the plot title, axis labels, and legend
    plt.title('Training and Validation PCC')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson')
    plt.ylim([0.15, 0.4])
    plt.legend()

    # Show the plot
    plt.savefig(save_path+'/pcc_plt.png')



def save_result_csv(mode_name, epoch, video_names, output, target):

    head = ['video name', 'Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
    
    save_path = './fold_Results/' + mode_name

    
    predict_save_path = save_path + '/epoch_' + str(epoch) +'_predicted.csv'
    target_save_path = save_path + '/label.csv'
    
    with open(predict_save_path, 'w',newline='') as f: 
        wr = csv.writer(f) 
        results = [[video_names[i].split('/')[-1]]+output[i] for i in range(len(video_names))]
        
        wr.writerow(head)
        wr.writerows(results)
        
    with open(target_save_path, 'w',newline='') as f: 
        wr = csv.writer(f) 
        results = [[video_names[i].split('/')[-1]]+target[i] for i in range(len(video_names))]
        
        wr.writerow(head)
        wr.writerows(results)


