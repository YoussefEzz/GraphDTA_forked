from training_validation import fit, plot_errorevolution
from create_data import create_pytorch_data
import os
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

def Train(dataset = 'urv', model_type = GCNNet, cuda_name = "cuda:0", TRAIN_BATCH_SIZE = 512, TEST_BATCH_SIZE = 512, LR = 0.0005, NUM_EPOCHS = 50):
   
   # 
   print("creating pytorch data.")
   create_pytorch_data(dataset)
   print("pytorch data created.")
   trained_model, training_mse_list, validation_mse_list = fit(dataset, model_type, cuda_name, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LR ,NUM_EPOCHS)
   print(trained_model)
   plot_errorevolution(training_mse_list, validation_mse_list, model_type.__class__.__name__, dataset)
   return trained_model