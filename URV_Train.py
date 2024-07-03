from training_validation import fit, plot_errorevolution, subplots_errorevolution
from create_data import create_pytorch_data
import os
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

def Train(dataset = 'urv', model_type = GCNNet, cuda_name = "cuda:0", TRAIN_BATCH_SIZE = 512, TEST_BATCH_SIZE = 512, LR = 0.0005, NUM_EPOCHS = 50, plot = True, overwrite = True, validation_size= 0.2):
   
   # 
   print("creating pytorch data.")
   create_pytorch_data(dataset, overwrite = overwrite)
   print("pytorch data created.")

   trained_model, training_mse_list, validation_mse_list, best_epoch = fit(dataset, model_type, cuda_name, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LR = LR, validation_size = validation_size ,NUM_EPOCHS = NUM_EPOCHS)

   print(trained_model)
   if(plot == True):
      plot_errorevolution(training_mse_list, validation_mse_list, model_type.__class__.__name__, dataset, best_epoch)
      return trained_model
   else:
      return trained_model, training_mse_list, validation_mse_list
  
      
def Trainfold(dataset = 'urv', model_type = GCNNet, cuda_name = "cuda:0", TRAIN_BATCH_SIZE = 512, TEST_BATCH_SIZE = 512, LR = 0.0005, NUM_EPOCHS = 50, plot = True, n_folds = 1, overwrite = True):
   trained_models = []
   list_training_mse_list = []
   list_validation_mse_list = []

   for i in range(1, n_folds + 1):

      dataset_fold = dataset + "_fold" + str(i)
      print("creating pytorch data.")
      create_pytorch_data(dataset_fold, overwrite = overwrite)
      print("pytorch data created.")

      trained_model, training_mse_list, validation_mse_list = fit(dataset_fold, model_type, cuda_name, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LR =  LR, validation_size= 0.2 ,NUM_EPOCHS = NUM_EPOCHS)
      trained_models.append(trained_model)
      list_training_mse_list.append(training_mse_list)
      list_validation_mse_list.append(validation_mse_list)

   if(plot == True):
      subplots_errorevolution(list_training_mse_list, list_validation_mse_list, dataset, n_folds)
      return trained_model
   else:
      return trained_model, list_training_mse_list, list_validation_mse_list