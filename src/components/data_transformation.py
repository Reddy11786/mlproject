import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl") 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    #coverts cat to num
    def get_data_transformer_object(self):
        '''
        func is used for data transformation
        '''
        try:
           numerical_columns=  ['reading_score', 'writing_score']
           
           catgorical_columns =['gender', 'race_ethnicity', 'parental_level_of_education', 
                                 'lunch', 'test_preparation_course']
           
           num_pipline = Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy="median")),
                   ("scaler",StandardScaler(with_mean=False))
               ]
           )
           cat_pipeline=Pipeline(
                   
                   steps=[
                       ("imputer",SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder",OneHotEncoder()),
                        ("scaler",StandardScaler(with_mean=False))
                   ]
            )
           logging.info(f"numerical col scaler done:{numerical_columns}")
           logging.info(f"categorical col encoding done:{catgorical_columns}")
           
           
           preprocessor= ColumnTransformer(
               [("num_pipeline",num_pipline,numerical_columns),
                ("cat_pipeline",cat_pipeline,catgorical_columns)]
               
           )
           
           return preprocessor
       
        except Exception as e:
            return CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("train and test csv file read")
            logging.info("obtaining preprocessing obj")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column= 'math_score'
            numerical_colums= ['writing_score','reading_score']
            
            input_feature_train_df =train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df =test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"applying preprocessing obj on train df and testing df")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                
            ]
            
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing obj")
            
            save_object(
                
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            
            return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            return CustomException(e,sys)
    
    
   
        
    
    
    
    
    
    
