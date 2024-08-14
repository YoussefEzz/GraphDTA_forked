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
from matplotlib.table import Table

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
    model = model.to(device)
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # print(f"Input data device: {data.x.device}")
            # print(f"Model device: {next(model.parameters()).device}")
            output = model(data)
            
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
    return total_labels.cpu().numpy().flatten(),total_preds.cpu().numpy().flatten()

LOG_INTERVAL = 20

def fit(dataset, model, cuda_name, TRAIN_BATCH_SIZE = 512, TEST_BATCH_SIZE = 512, LR = 0.0005, validation_size = 0.2, NUM_EPOCHS = 50, best_model_flag = True):
    # datasets = [['davis','kiba','urv'][int(sys.argv[1])]] 
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
    print('\nrunning on ', model_st + '_' + dataset )
    print('\nrunning on dataset : ', dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', overwrite= False)
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', overwrite= False)
        
        
        #train_size = int(0.8 * len(train_data))
        train_size = int((1.0 - validation_size) * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
        
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        best_model = model
        best_validation_mse = 1000
        best_epoch = -1

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_test_mse = 1000
        training_mse_list = []
        validation_mse_list = []
        best_test_ci = 0
        
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'

        
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, loss_fn)
            training_mse_list.append(train_loss.item())
            print('predicting for valid data')
            G,P = predicting(model, device, valid_loader)
            validation_mse = mse(G,P)
            validation_mse_list.append(validation_mse)
            

            if best_model_flag == True:
                if validation_mse < best_validation_mse:
                    best_validation_mse = validation_mse
                    best_model = model
                    best_epoch = epoch+1
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
        if best_model_flag == True:
            torch.save(best_model.state_dict(), model_file_name)
        else:
            torch.save(model.state_dict(), model_file_name)
    if best_model_flag == True:
        return best_model, training_mse_list, validation_mse_list, best_epoch, train_size, valid_size
    else:
        return model, training_mse_list, validation_mse_list, train_size, valid_size
            

# Scatter plot. Vertical axis: predicted value. Horizontal axis: real value
def scatterplot(real_values, predicted_values, model_name, dataset):
    x_line = [0, 11]
    y_line = [0, 11]
    # Points for the diagonal line
    if "davis" in dataset:
        x_line = [0, 11]
        y_line = [0, 11]
    elif "kiba" in dataset:
        x_line = [0, 18]
        y_line = [0, 18]
    else: 
        x_line = [0, 11]
        y_line = [0, 11]
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

# subplots of Scatter plot. Vertical axis: predicted value. Horizontal axis: real value
def subplots_scatterplot(real_values, predicted_values, mse_list, model_name, dataset, n_folds):

    # Create a figure
    fig, axs = plt.subplots(n_folds, 1, figsize=(24, 4*n_folds))  # n_folds rows, 1 columns
    for i in range(0, n_folds):
        # Points for the diagonal line
        x_line = [0, 8]
        y_line = [0, 8]
        axs[i].scatter(real_values[i], predicted_values[i], color = 'red')
        # Plot the line
        axs[i].plot(x_line, y_line, color='blue', linestyle='--')

        axs[i].set_xlabel('real affinity')
        axs[i].set_ylabel('predicted affinity')
        axs[i].set_title(f'real vs predicted affinity for model {model_name} on dataset {dataset} fold {i + 1} with MSE {mse_list[i]:.2f}')
        # Set the aspect ratio of the subplot to 'equal' to make it a square
        axs[i].set_aspect('equal')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

# Plot the evolution of the training and validation loss with number of epochs
def plot_errorevolution(training_mse_list, validation_mse_list, model_name, dataset, best_epoch, LR, NUM_EPOCHS, TRAIN_BATCH_SIZE, validation_size, train_size, valid_size):
    plt.figure(figsize=(20, 6))
    # Plot the evolution of the training and validation loss
    epochs_training_list = list(range(1, len(training_mse_list) + 1))
    epochs_validation_list = list(range(1, len(validation_mse_list) + 1))
    plt.plot(epochs_training_list, training_mse_list,  label='Training Loss')
    plt.plot(epochs_validation_list, validation_mse_list,  label='Validation Loss')
    plt.axvline(x = best_epoch, color='r', linestyle='--', label='best model')

    # Add the training parameters outside the plot
    # Parameters to display in the table
    parameters = {
        "optimizer": "ADAM",
        "learning rate": LR,
        "epochs": NUM_EPOCHS,
        "train batch size" : TRAIN_BATCH_SIZE,
        "train size" : train_size,
        "validation size": valid_size,
        "validation percentage": str(validation_size * 100) + ' %',
        "MSE": validation_mse_list[best_epoch - 1]
    }

    # Create the table and add it to the plot
    table_data = [[key, value] for key, value in parameters.items()]
    table = plt.table(cellText=table_data, loc='right', cellLoc='center', colLoc='center', bbox=[1.2, 0.1, 0.3, 0.8])

    # Customize the table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1)

    plt.title(f'evolution of the mean square error for model {model_name} on database : {dataset}')
    plt.legend( bbox_to_anchor=(1, 1), ncol=1, fontsize=12)
    plt.subplots_adjust(right=0.75)
    plt.show()

# subplot the evolution of the training and validation loss with number of epochs f for k folds
def subplots_errorevolution(list_training_mse_list, list_validation_mse_list, model_name, dataset, n_folds, list_best_epochs, LR, NUM_EPOCHS, TRAIN_BATCH_SIZE, validation_size, list_train_size, list_valid_size):

    # Create a figure
    fig, axs = plt.subplots(nrows=n_folds, ncols=1, figsize=(24, 4*n_folds))  # n_folds rows, 1 columns
    epochs_list = list(range(1, len(list_training_mse_list[0]) + 1))
    # print(epochs_list)
    # print(list_training_mse_list[0])
    for i in range(0, n_folds):

        epochs_training_list = list(range(1, len(list_training_mse_list[i]) + 1))
        epochs_validation_list = list(range(1, len(list_validation_mse_list[i]) + 1))
        # Plot data on the first subplot
        axs[i].plot(epochs_training_list, list_training_mse_list[i], label='Training Loss')
        axs[i].plot(epochs_validation_list, list_validation_mse_list[i], label='Validation Loss')
        axs[i].axvline(x = list_best_epochs[i], color='r', linestyle='--', label='best model')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('mean square error')
        axs[i].set_title(f'MSE for {dataset} fold {i + 1}')
        axs[i].legend()

        # Add the training parameters outside the plot
        # Parameters to display in the table
        parameters = {
            "fold": i + 1,
            "optimizer": "ADAM",
            "learning rate": LR,
            "epochs": NUM_EPOCHS,
            "train batch size" : TRAIN_BATCH_SIZE,
            "train size" : list_train_size[i],
            "validation size": list_valid_size[i],
            "validation percentage": str(validation_size * 100) + ' %',
            "MSE": list_validation_mse_list[i][list_best_epochs[i] - 1]
        }

        # Create the table and add it to the plot
        table_data = [[key, value] for key, value in parameters.items()]
        table = axs[i].table(cellText=table_data, loc='right', cellLoc='center', colLoc='center', bbox=[1.2, 0.1, 0.3, 0.8])

        # Customize the table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()