from training_validation import predicting, TestbedDataset, DataLoader, scatterplot
import os
import torch
from utils import mse
def Test(trained_model, dataset = 'urv', cuda_name = "cuda:0", TEST_BATCH_SIZE = 512):
    print('predicting for test data')

    test_data = TestbedDataset(root='data', dataset=dataset+'_test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    G,P = predicting(trained_model, device, test_loader)
    scatterplot(G, P, trained_model.__class__.__name__, dataset)
    return mse(G,P)
    
    