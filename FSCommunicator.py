
import ipfshttpclient
import os

import shutil
import numpy as np
import joblib
import io
# create an empty list to store all the model hashes
model_hashes =[]

class FSCommunicator:
    def __init__(self, ipfs_path):
        self.ipfs_path = ipfs_path


        # Connect to the IPFS daemon
        self.client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
        files = self.client.ls(self.ipfs_path)

    def fetch_initial_model(self):
        # Load the model and optimizer state_dicts from IPFS

        model_hash = 'QmP7yKRDPDLzNnDWTUfw3JGJP2i1oay4bxViWDgmPL27Mj'
        model_bytes = self.client.cat(model_hash)
        model = joblib.load(io.BytesIO(model_bytes))
        print("Done Model!!!!!!")


        return model
    
    def fetch_evaluation_models(self, worker_index, round, num_workers):
        print("Fetch Function")
        state_dicts = []
        
        
        for i in range(num_workers):
            print("worker",i)
            print("hash length",len(model_hashes))
            if i < (len(model_hashes)-1):
                model_hash = model_hashes[i]
                model_bytes = self.client.cat(model_hash)
                print(model_hash)
                state_dicts.append(joblib.load(io.BytesIO(model_bytes)))
        # print("Fetch state dicts", state_dicts)
        
      
        return state_dicts  
    
    def push_model(self, state_dict, worker_index, round, num_workers):
        
        # Clear the model_hashes list if it has reached the number of workers

        if (len(model_hashes)) == num_workers:
                model_hashes.clear()
                print("model_hashes list cleared",model_hashes)
                
        # Save the state_dict to a file
        print("Pushing Model")
        model_filename = 'model_round_{}_index_{}.joblib'.format(round, worker_index)
        joblib.dump(state_dict, model_filename)
        print("MODEL SAVE TO LOCAL")

        # Add the file to IPFS and get the new hash
        model_has = self.client.add(model_filename)
        model_hash = model_has['Hash']
       

        print("Length of model hashes:", len(model_hashes))
        

        model_hashes.append(model_hash)  # add the new model hash to the list
        print("List of hash:", model_hashes)
        print("Model Hash:", model_hash)
        print("Pushing Complete")
        # print(loaded_model)
        # Remove the local file
        # os.remove(model_filename)

        

        