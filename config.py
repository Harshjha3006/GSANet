import argparse
import torch


class Config:
    def __init__(self,**kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k,v in kwargs.items():
            setattr(self,k,v)
        

def get_config(**optional_kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_type",type = str,default = "TVSum",help = "choose the type of the dataset [TVSum|SumMe]")
        parser.add_argument("--batch_size",type = int,default = 1)
        parser.add_argument("--split_index",type = int,default = 0,help ="which split to run the model on")
        parser.add_argument("--embed_dim",type = int,default=1024,help = "dimension of feature embeddings")
        parser.add_argument("--seed",type = int,default = 12345,help= "chosen seed for generating random numbers")
        parser.add_argument("--clip",type = float,default=5.0,help = "Max Norm of the gradients")
        parser.add_argument("--num_heads",type = int,default=8)
        parser.add_argument("--num_segments",type = int,default = 10)
        parser.add_argument("--dropout_rate",type = float,default=0.2)
        parser.add_argument("--hidden_dim1",type = int,default=256)
        parser.add_argument("--hidden_dim2",type = int,default=128)
        parser.add_argument("--final_dim",type = int,default=1)
        parser.add_argument("--threshold",type = int,default=10)
        parser.add_argument("--num_layers",type = int,default=6)
        parser.add_argument("--num_epochs",type = int,default= 100)
        parser.add_argument("--lr",type = float,default=1e-5)
        parser.add_argument("--l2_reg",type = float,default=1e-4)
        parser.add_argument("--all_splits",action = 'store_true')
        parser.add_argument("--exp_name",type = str,default = "default_exp")
        args,_ = parser.parse_known_args()
        kwargs = vars(args)
        kwargs.update(optional_kwargs)
        return Config(**kwargs)
    
