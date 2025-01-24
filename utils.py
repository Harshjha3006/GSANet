from scipy.stats import spearmanr,kendalltau,rankdata
import numpy as np
import torch.nn as nn

def compute_rho_tau(model_scores,ref_scores):

    if model_scores.shape[0] < ref_scores.shape[0]:
        ref_scores = ref_scores[:model_scores.shape[0]]

    rho = spearmanr(model_scores,ref_scores)[0]
    tau = kendalltau(-rankdata(model_scores),-rankdata(ref_scores))[0]

    return rho,tau



def compute_frame_scores(preds,n_frames):

    # initialise the result array 
    frame_scores = np.zeros((n_frames))


    for pos in range(n_frames):
        if pos >= preds.shape[0]:
            frame_scores[pos] = 0
        else:
            frame_scores[pos] = preds[pos]

    return frame_scores



def compute_seg_scores(frame_scores,cps):

    num_segs = cps.shape[0]

    seg_scores = []

    for i in range(num_segs):

        start_pos = cps[i][0]
        end_pos = cps[i][1] + 1

        seg_scores.append(np.mean(frame_scores[start_pos : end_pos]))


    return seg_scores




def knapsack(capacity, weights, values, num_items):

    # K[i][w] = max value out of i items with w max capacity 

    K = [[0 for _ in range(capacity + 1)] for _ in range(num_items + 1)]

    for i in range(num_items + 1):
        for w in range(capacity + 1): 
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i - 1] <= w:
                K[i][w] = max(values[i - 1] + K[i - 1][w - weights[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = capacity
    for i in range(num_items,0,-1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0,i - 1)
            w -= weights[i - 1]
    return selected



def build_summary(n_frames, cps, selected_segs):
    
    summary = np.zeros(n_frames)
    num_segs = cps.shape[0]

    for seg in range(num_segs):
        if seg in selected_segs:
            start_pos = cps[seg][0]
            end_pos = cps[seg][1] + 1

            summary[start_pos : end_pos] = 1

    return summary 

def compute_f1score(machine_summary,user_summaries,dataset_type):

    f1_scores = []
    num_users = user_summaries.shape[0]

    machine_summary = machine_summary.astype(np.int64)
    user_summaries = user_summaries.astype(np.int64)

    for i in range(num_users): 
        user_summary = user_summaries[i,:]
        if len(user_summary) > len(machine_summary):
            user_summary = user_summary[:len(machine_summary)]
        overlap = np.bitwise_and(machine_summary,user_summary).sum()
        precision = overlap / (machine_summary.sum() + 1e-8)
        recall = overlap / (user_summary.sum() + 1e-8)

        if precision == 0 or recall == 0:
            f1_score = 0
        else: 
            f1_score = (2 * precision * recall) / (precision + recall)
        
        f1_scores.append(f1_score)

    if dataset_type == 'TVSum': 
        return np.mean(f1_scores)
    else:
        return np.max(f1_scores)
    


def init_weights(model):
    for name, param in model.mhaList.named_parameters():
            if 'weight' in name and "norm" not in name:
                if param.dim() > 1 :
                    nn.init.kaiming_uniform_(param,mode = "fan_in",nonlinearity="relu")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

