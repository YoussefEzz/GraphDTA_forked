from training_validation import predicting, TestbedDataset, DataLoader, scatterplot, subplots_scatterplot
import os
import torch
from utils import mse

def Test(trained_model, dataset = 'urv', cuda_name = "cuda:0", TEST_BATCH_SIZE = 512, plot = True, n_folds = 1):
    print('predicting for test data')

    test_data = TestbedDataset(root='data', dataset=dataset+'_test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    G,P = predicting(trained_model, device, test_loader)
    if(plot == True):
        scatterplot(G, P, trained_model.__class__.__name__, dataset)
    return mse(G,P)
        
        
def Testfold(model, dataset = 'urv', cuda_name = "cuda:0", TEST_BATCH_SIZE = 512, plot = True, n_folds = 10):
    
    realvalues_list = []
    predictedvalues_list = []
    mse_list = []
    for i in range(1, n_folds + 1):

        # Load the state dictionary from the file
        model_file_name = 'model_' + model.__class__.__name__ + '_' + dataset + '_fold' + str(i) + '.model'
        state_dict = torch.load(model_file_name)

        trained_model = model
        # Load the state dictionary into the model
        trained_model.load_state_dict(state_dict)

        # Optionally, set the model to evaluation mode
        trained_model.eval()    

        print('predicting for test data')
        test_data = TestbedDataset(root='data', dataset=dataset + '_fold' + str(i) + '_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        G,P = predicting(trained_model, device, test_loader)
        
        realvalues_list.append(G)
        predictedvalues_list.append(P)
        mse_list.append(mse(G,P))

    if(plot == True):
        subplots_scatterplot(realvalues_list, predictedvalues_list, mse_list, trained_model.__class__.__name__, dataset, n_folds)
    return mse_list