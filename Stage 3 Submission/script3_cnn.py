from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- Multi-Layer Perceptron script for 10 Labels ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('Stage 3 Data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    #data_obj.dataset_source_file_name = 'ORL'
    #data_obj.dataset_source_file_name = 'MNIST'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN('Convolutional layers', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/Stage3_CNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('Evaluation metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, recall, f1score = setting_obj.load_run_save_evaluate()
    print('************ Test Set Performance ************')
    print('CNN -> Accuracy: ' + str(accuracy) +
          'Precision: ' + str(precision) +
          'Recall: ' + str(recall) +
          'F1 Score: ' + str(f1score))

    print('\n************ Training Plots ************')
    epochs = range(1, len(method_obj.loss) + 1)


    evaluate_obj.setPlotData(epochs, method_obj.loss, 'loss')
    evaluate_obj.setPlotData(epochs, evaluate_obj.accuracy, 'accuracy')
    evaluate_obj.setPlotData(epochs, evaluate_obj.precision, 'precision')
    evaluate_obj.setPlotData(epochs, evaluate_obj.recall, 'recall')
    evaluate_obj.setPlotData(epochs, evaluate_obj.f1score, 'f1 score')

    evaluate_obj.setPlotLabels('Epochs', 'Metrics', 'Metrics over Epochs')
    evaluate_obj.showPlot()
    print('************ Finish ************')
    # ------------------------------------------------------
    

    