'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_2_code.Dataset_Loader import Dataset_Loader
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data_train = self.dataset.load()

        test_data_obj = Dataset_Loader('Multiclass test', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
        test_data_obj.dataset_source_file_name = 'test.csv'

        loaded_data_test = test_data_obj.load()


        #y_train = loaded_data_train[:,0]
        #X_train = loaded_data_train[:,1:]

        #y_test = loaded_data_test[:,0]
        #X_test = loaded_data_test[:,1:]

        # run MethodModule
        self.method.data = {'train': loaded_data_train, 'test': loaded_data_test}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        