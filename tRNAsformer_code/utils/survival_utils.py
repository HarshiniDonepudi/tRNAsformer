import numpy as np
import torch
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))

def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)