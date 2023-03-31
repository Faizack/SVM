import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

class Model():
    '''
    Contains all machine learning functionality
    '''
    # static, might have to be calculated dynamically
    batch_size = 64
    epochs = 1

    def __init__(self, num_workers, idx, model, optimizer, topk, isEvil = False):
        self.num_workers = num_workers
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.topk = topk
        self.isEvil = isEvil
        
        data=pd.read_csv('train.csv')
        #this would be generic in a real application
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
          # Split the dataset into training and testing sets
        xtrain_all, xtest_all, ytrain_all, ytest_all = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Calculate the size of the data subset for each worker
        subset_size = len(xtrain_all) // num_workers
        
        # Calculate the start and end indices of the data subset for the current worker
        start_idx = idx * subset_size
        end_idx = (idx + 1) * subset_size
        
        # Use the start and end indices to select the data subset for the current worker
        data_subset = X[start_idx:end_idx], y[start_idx:end_idx]
        
        # Use the start and end indices to select the data subset for the current worker
        self.xtrain = xtrain_all[start_idx:end_idx]
        self.ytrain = ytrain_all[start_idx:end_idx]
        self.xtest = pd.read_csv('test.csv').values
        self.ytest = ytest_all

        
        
    def average(self,state_dicts):
        print("Averaging")
        super_Lr = LogisticRegression()

        coefs = np.array([model.coef_ for model in state_dicts])
        intercepts = np.array([model.intercept_ for model in state_dicts])
        super_Lr.coef_ = np.mean(coefs, axis=0)
        super_Lr.intercept_ = np.mean(intercepts, axis=0)
        print(f"Super Model Coeffecient : {self.model.coef_}")
        print(f"Super Model Intercepter : {self.model.intercept_}")
        return super_Lr
    
    
    def adapt_current_model(self, avg_state_dict):
        self.model=avg_state_dict


    def train(self):
        print("Traning")
        
        for epoch in range(self.epochs):
            self.model.fit(self.xtrain, self.ytrain)
            print(f"Coeffecient : {self.model.coef_}")
            print(f"Intercepter : {self.model.intercept_}")
            print(f'Finished epoch {epoch}')
        print("Training  Model ",self.model)
        return self.model


    def rank_models(self, sorted_models):
        return [self.num_workers - idx for idx in range(len(sorted_models))]
    
    def get_top_k(self, sorted_models):
        return [models[2] for models in sorted_models][-self.topk:]
    
    def test(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.xtest)

        # Evaluate the model's accuracy
        accuracy = accuracy_score(self.ytest, y_pred)
        print(" Test Accuracy: ", accuracy)
        return accuracy
    
    def eval(self, model_state_dicts):
        res = []
        for idx, m in enumerate(model_state_dicts):
            self.model=m
            acc = self.test()
            res.append((acc,idx,m))
            print("Res",res)
        sorted_models = sorted(res, key=lambda t: t[0])
        return self.rank_models(sorted_models),  self.get_top_k(sorted_models), res
            
            
            
            
        
        
        
