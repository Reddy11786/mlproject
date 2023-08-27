import os
import sys
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from src.utils import save_object
from sklearn.linear_model import Lasso,Ridge
from src.utils import evaluate_models
#from xgboost import XGBRegressor
#from catboost import CatBoostRegressor



@dataclass 
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config= ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                                             )
            
            models= {
                
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                #"XGBRegressor": XGBRegressor(),
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "RandomForestRegressor":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                 
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                 "Lasso": {}, 
                 "Ridge": {},  
                 "KNeighbors Regressor": {},  
               
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models,param=params)
            
            best_model_score =max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("best model not there")
            logging.info(f"best model found on both train n test dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)