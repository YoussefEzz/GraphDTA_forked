import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import matplotlib.pyplot as plt

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss

# equivalent to feedforward
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

LOG_INTERVAL = 20

def fit(dataset, model, cuda_name, TRAIN_BATCH_SIZE = 512, TEST_BATCH_SIZE = 512, LR = 0.0005, NUM_EPOCHS = 50):
    # datasets = [['davis','kiba','urv'][int(sys.argv[1])]] 
    datasets = [dataset]
    # modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
    modeling = model
    model_st = modeling.__class__.__name__
    print('model_st:', model_st)
    # cuda_name = "cuda:0"
    # if len(sys.argv)>3:
    #     cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)

    # TEST_BATCH_SIZE = 512
    # LR = 0.0005

    # NUM_EPOCHS = 50

    # print('Learning rate: ', LR)
    # print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    for dataset in datasets:
        print('\nrunning on ', model_st + '_' + dataset )
        print('\nrunning on dataset : ', dataset )
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if (((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))) and (dataset != 'urv')):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test')
            
            
            train_size = int(0.8 * len(train_data))
            valid_size = len(train_data) - train_size
            train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
            
            
            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_test_mse = 1000
            training_mse_list = []
            validation_mse_list = []
            best_test_ci = 0
            best_epoch = -1
            model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train_loss = train(model, device, train_loader, optimizer, epoch+1, loss_fn)
                training_mse_list.append(train_loss.item())
                print('predicting for valid data')
                G,P = predicting(model, device, valid_loader)
                val = mse(G,P)
                validation_mse_list.append(val)
                # if val<best_mse:
                #     best_mse = val
                #     best_epoch = epoch+1
                #     torch.save(model.state_dict(), model_file_name)
                #     print('predicting for test data')
                #     G,P = predicting(model, device, test_loader)
                #     ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                #     with open(result_file_name,'w') as f:
                #         f.write(','.join(map(str,ret)))
                #     best_test_mse = ret[1]
                #     best_test_ci = ret[-1]
                #     print('rmse improved at epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)
                # else:
                #     print(ret[1],'No improvement since epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)
            torch.save(model.state_dict(), model_file_name)
    return model, training_mse_list, validation_mse_list
            

# Scatter plot. Vertical axis: predicted value. Horizontal axis: real value
def scatterplot(real_values, predicted_values, model_name, dataset):

    # Points for the diagonal line
    x_line = [0, 8]
    y_line = [0, 8]
    plt.scatter(real_values, predicted_values, color = 'red')
    # Plot the line
    plt.plot(x_line, y_line, color='blue', linestyle='--')

    plt.xlabel('real affinity')
    plt.ylabel('predicted affinity')
    plt.title(f'real affinity vs predicted affinity for model {model_name} on database {dataset}')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

    # # Adjust layout to make room for the legend
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
# Plot the evolution of the training and validation loss with number of epochs
def plot_errorevolution(training_mse_list, validation_mse_list, model_name, dataset):
    # Plot the evolution of the training and validation loss
    epochs_training_list = list(range(1, len(training_mse_list) + 1))
    epochs_validation_list = list(range(1, len(validation_mse_list) + 1))
    plt.plot(epochs_training_list, training_mse_list,  label='Training Loss')
    plt.plot(epochs_validation_list, validation_mse_list,  label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('mean square error')
    plt.title(f'evolution of the mean square error for model {model_name} on database : {dataset}')
    plt.legend()
    plt.show()
