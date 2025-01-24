import math
import torch 
import numpy as np
import os
from config import get_config
from dataloader import get_data_loader
from utils import build_summary, compute_f1score, compute_frame_scores, compute_seg_scores, init_weights, knapsack
import h5py
from model.summaryNet import SummaryNet
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
from dataloader import DATASET_PATH
import random

checkpoint_path = "checkpoints"
results_path = "results"
logs_path = "tensorboard_logs"


class Trainer:
    def __init__(self,config):

        self.config = config
        self.train_loader,self.test_loader = get_data_loader(self.config)
        self.model = None
        self.optimizer = None
        self.logger = None

        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        self.criterion = nn.CrossEntropyLoss()


    def train_all(self):

        if os.path.exists(results_path) == False:
            os.makedirs(results_path)

        results = []
        for split_index in range(0,5):
            print(f"Starting training for split {split_index}\n")
            self.config.split_index = split_index
            best_epoch,best_fscore = self.train()
            results.append({"best_epoch" : best_epoch,"best_fscore" : best_fscore})
            print()
        fscores = []
        for i,result in enumerate(results):
            print(f"""Split {i}, Best Epoch(as per F1 Score) : {result["best_epoch"]},Best F1 Score : {result["best_fscore"]}""")
            print()
            with open(f"{results_path}/{self.config.exp_name}.txt","a") as file:
                file.write(f"""Split {i}, Best Epoch(as per F1 Score) : {result["best_epoch"]},Best F1 Score : {result["best_fscore"]}\n""")

            fscores.append(result["best_fscore"])
        max_fscore,mean_fscore = np.array(fscores).max(),np.array(fscores).mean()

        print(f"Max F1-score: {max_fscore},Mean F1 score : {mean_fscore}")
        with open(f"{results_path}/{self.config.exp_name}.txt","a") as file:
            file.write(f"Max F1-score: {max_fscore},Mean F1 score : {mean_fscore}\n")



    def train(self):    

        if os.path.exists(results_path) == False:
            os.makedirs(results_path)

        # initalise instance variables

        self.model = SummaryNet(self.config).to(self.config.device)
        init_weights(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.config.lr,weight_decay=self.config.l2_reg)

        self.logger = SummaryWriter(log_dir=f"{logs_path}/{self.config.exp_name}_{self.config.dataset_type}_split{self.config.split_index}")


        # initialize variables for holding best metrics and state of the model
        best_epoch = 0,
        best_f1score = 0
        best_checkpoint = {}


        # start the training by iterating over all epochs 

        for epoch in range(self.config.num_epochs):

            # set the model to training mode 
        
            self.model.train()

            # intialise loss_history list for computing avg loss of every epoch 

            loss_history = []

            # iterate over all batches

            iterator = iter(self.train_loader)
            num_batches = int(len(self.train_loader) / self.config.batch_size)
            for _ in range(num_batches):
                # reset the optimzier so that it will calculate gradients over the whole batch 

                self.optimizer.zero_grad()

                for _ in range(self.config.batch_size):

                    batch = next(iterator)

                    # Extract the inputs and targets from the batch and transfer them to the appropriate device 

                    features,targets = batch
                    x = features.to(self.config.device)
                    y = targets.to(self.config.device).to(torch.long)  

                    # Feed the inputs and outputs(so that it gets clipped)
                    preds,y = self.model(x,y)

                    # Compute the loss and update loss_history
                    # preds reshaped to (b,2,num_frames) , y reshaped to (b,num_frames)
                    loss = self.criterion(preds.permute(0,2,1),y.squeeze(-1))

                    loss_history.append(loss.item())

                    # Compute gradients

                    loss.backward()

                # update the weights
                self.optimizer.step()


            # Compute evaluation metrics 
            # switch the model to evaluation mode
            self.model.eval()

            # compute f1score, rho , tau and avg loss

            f1_score = self.evaluate()
            avg_loss = np.mean(loss_history).item()
            print(f"Epoch {epoch + 1}, avg loss : {avg_loss}, F1 Score : {f1_score}")

            # update the global best metric

            if f1_score > best_f1score : 
                    best_epoch = epoch + 1
                    best_f1score = f1_score
                    best_checkpoint = {
                        "epoch" : epoch + 1,
                        "model_state" : self.model.state_dict(),
                        "optimizer_state" : self.optimizer.state_dict(),
                        "config" : self.config
                    }

            # Log metrics to tensorboard

            self.logger.add_scalar("Loss",avg_loss,epoch + 1)
            self.logger.add_scalar("F1 Score",f1_score,epoch + 1)


        # Close the logger
    
        self.logger.close()

        # Save the checkpoint to a pth file

        if os.path.exists(checkpoint_path) == False:
            os.makedirs(checkpoint_path)

        torch.save(best_checkpoint,f"{checkpoint_path}/{self.config.dataset_type}_split{self.config.split_index}_epoch{best_epoch}.pth")

        # Report and Return the Best Metrics 

        print(f"Best Epoch : {best_epoch}")
        print(f"Best F1 Score : {best_f1score}")

        with open(f"{results_path}/{self.config.exp_name}.txt","a") as file:
            file.write(f"Best Epoch : {best_epoch}\n")
            file.write(f"Best F1 Score : {best_f1score}\n")

        return best_epoch,best_f1score
    

        

    def evaluate(self):

        # intialise h5py dataset

        h5file = None 

        if self.config.dataset_type == "TVSum":
            h5file = h5py.File(f"{DATASET_PATH}/TVSum.h5",'r')
        else:
            h5file = h5py.File(f"{DATASET_PATH}/SumMe.h5",'r')
    
        # Initialise the lists that will hold the metrics for every video in the test set 

        f1_scores = []


        # iterate over every batch by keeping torch no_grad off 

        with torch.no_grad():


            for video_data in self.test_loader:

                # extract the features and targets and transfer them to the gpu 

                features,targets,video_ind = video_data
                x = features.to(self.config.device)
                y = targets.to(self.config.device)
                video_ind = video_ind.item()
                # compute the frame importance scores from the model

                preds,y = self.model(x,y)

                # detach the preds and targets from torch,transfer them to cpu and turn them into numpy arrays 
                
                sigmoid = nn.Sigmoid()
                preds= sigmoid(preds)[:,:,1]

                preds = preds.squeeze(0).detach().cpu().numpy()
                y = y.squeeze(0).detach().cpu().numpy()


                # extract other data required for computation of f1 score
                # change points for obtaining segment endpoints obtained through KTS
                cps = h5file[f"video_{video_ind}"]["change_points"][...]
                # total number of frames in the video 
                n_frames = h5file[f"video_{video_ind}"]["length"][()].item()
                # user summaries for computing f1 scores 
                user_summary = h5file[f"video_{video_ind}"]["user_summary"][...]
                # number of frames per segment
                nfps = h5file[f"video_{video_ind}"]["n_frame_per_seg"][...]

                # Compute F1 score 

                # extraplate imp scores of sampled frames to all frames 

                frame_scores = compute_frame_scores(preds,n_frames)

                # compute seg scores 

                seg_scores = compute_seg_scores(frame_scores,cps)

                # compute max allowable length 

                max_length = int(math.floor(n_frames * 0.15))

                # use knapsack to get optimal segments

                selected_segs = knapsack(capacity = max_length,weights = nfps, values = seg_scores,num_items = len(seg_scores))

                # build summary 

                summary = build_summary(n_frames,cps,selected_segs)

                # compute f1 score with user summaries

                f1_score = compute_f1score(summary, user_summary,self.config.dataset_type)

                f1_scores.append(f1_score)

        h5file.close()
    
        return np.mean(f1_scores)


                


if __name__ == "__main__":
    config = get_config()
    trainer = Trainer(config)
    if config.all_splits:
        trainer.train_all()
    else:
        trainer.train()



            

        

