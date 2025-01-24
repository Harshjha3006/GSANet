import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import json
import numpy as np
import yaml

DATASET_PATH = "datasets"
SPLIT_PATH = "splits"


class VideoFramesData(Dataset):
    def __init__(self,mode,dataset_type,split_index):
        self.datasets = [f"{DATASET_PATH}/SumMe.h5",
                         f"{DATASET_PATH}/TVSum.h5"]
        
        self.split_files = [f"{SPLIT_PATH}/summe.yml",f"{SPLIT_PATH}/tvsum.yml"]

        self.dataset,self.split_file,self.split = None,None,None
        self.mode = mode 

        if dataset_type == "TVSum":
            self.dataset = self.datasets[1]
            self.split_file = self.split_files[1]
        else:
            self.dataset = self.datasets[0]
            self.split_file = self.split_files[0]

        h5file = h5py.File(self.dataset,'r')

        self.feature_list = []
        self.target_list = []
        self.index_list = []

        with open(self.split_file,'r') as file:
            data = yaml.safe_load(file)
            for i,split in enumerate(data):
                if i == split_index:
                    self.split = split
                    break
        for video_name in self.split[mode + "_keys"]:
            features = torch.tensor(np.array(h5file[video_name + "/feature"]),dtype = torch.float32)
            # for making the total dimension of (batch_size,seq_len,1)
            gtsummary = torch.tensor(np.array(h5file[video_name + "/label"]),dtype = torch.float32).unsqueeze(1)
            self.feature_list.append(features)
            self.target_list.append(gtsummary)
            video_ind = int(str(video_name).split('_')[-1])
            self.index_list.append(video_ind)
                                    
        h5file.close()

    def __len__(self):
        return len(self.feature_list)
    
    def __getitem__(self,index):
        if self.mode == "test":
            return self.feature_list[index],self.target_list[index],self.index_list[index]
        else:
            return self.feature_list[index],self.target_list[index]
    



def get_data_loader(config):
    trainvd = VideoFramesData("train",config.dataset_type,config.split_index)
    testvd = VideoFramesData("test",config.dataset_type,config.split_index)
    return  DataLoader(trainvd,batch_size = 1,shuffle=True),DataLoader(testvd,batch_size = 1,shuffle = True)