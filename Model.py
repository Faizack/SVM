import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

class Model():
    '''
    Contains all machine learning functionality
    '''
    # static, might have to be calculated dynamically
    batch_size = 64
    epochs = 3

    def __init__(self, num_workers, idx, model, optimizer, topk, isEvil = False):
        self.num_workers = num_workers
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.topk = topk
        self.isEvil = isEvil
        
        
        #this would be generic in a real application
        self.train_loader =  cancer.data

        self.test_loader = cancer.target
        

        
        
        # find the datasets indices
        # also this would not be implemented like this in the real application
        # the users would use an 80/20 random split of their own dataset for training/validating
        self.num_train_batches = len(self.train_loader)//self.num_workers
        self.num_test_batches = len(self.test_loader)//self.num_workers 
        # start idx
        self.start_idx_train = self.num_test_batches* self.idx
        self.start_idx_test = self.num_test_batches * self.idx
        
        
    def average(self, state_dicts):
        pass
    
    
    def adapt_current_model(self, avg_state_dict):
        self.model.load_state_dict(avg_state_dict)


    def train(self):
        
        for epoch in range(self.epochs):
            for idx, (data, target) in enumerate(self.train_loader):
                if idx >= self.start_idx_train and idx < self.start_idx_train + self.num_train_batches:
                    X_batch = self.train_loader
                    y_batch =self.test_loader

            # if is_evil:
            #     X_batch = garbage
            #     y_batch = np.random.randint(0, 10, batch_size)

                    self.model.fit(X_batch, y_batch)

            print(f'Finished epoch {epoch}')
        print("Training  Model ",self.model)
        return self.model


    def rank_models(self, sorted_models):
        return [self.num_workers - idx for idx in range(len(sorted_models))]
    
    def get_top_k(self, sorted_models):
        return [models[2] for models in sorted_models][-self.topk:]
    
    def test(self):
        pass
    
    def eval(self, model_state_dicts):
        res = []
        for idx, m in enumerate(model_state_dicts):
            self.model.load_state_dict(m)
            acc = self.test()
            res.append((acc,idx,m))
            print("Res",res)
        sorted_models = sorted(res, key=lambda t: t[0])
        return self.rank_models(sorted_models),  self.get_top_k(sorted_models), res
            
            
            
            
        
        
        
