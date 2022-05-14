import lib
import numpy as np
import torch
from tqdm import tqdm
import statistics

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        #start : precision was not calculated in original code
        precosions = []
        #end
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            #for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)#logit contain model scores 
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                #start
                recall, mrr ,precision = lib.evaluate(logit, target, k=self.topk)
                #end:
                       
                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                #start:
                losses.append(loss.item())
                #end:
                recalls.append(recall)
                precosions.append(precision)
                mrrs.append(mrr.item())
        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        #start: 
        mean_precision= np.mean(precosions) 
        #end:     
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_recall , mean_precision , mean_mrr