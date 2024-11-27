import torch
from tqdm import tqdm
import numpy as np
from src.visualize.visualize import generate_by_ori_video_sequences
import os

def train_or_test(model, optimizer, iterator, device, epoch, para, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in model.losses}
    batches = []
    gt = []
    pred = []
    
    with grad_env():
        for i, ori_batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            ori_batch = {key: val.to(device) for key, val in ori_batch.items()}

            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()
            
            '''
            print(ori_batch.keys())
            # forward pass
            if mode == 'test':
                print('yhat.....')
                print(ori_batch['y'])
            if mode == 'train':
                print('train...yhat.....')
                print(ori_batch['y'])
            
            '''
            '''
            print('888888888888888888888')
            if mode == 'train':
                
                print(ori_batch['x'].size())
                print(ori_batch['lengths'][0])
                #print(ori_batch['features'].size())
            '''

            #if i == 0 and mode == 'test':
            #    print(ori_batch['output_xyz'][0,10,2,3])
            batch = model(ori_batch)
            #if i == 0:
            #    print(batch['output_xyz'][0,10,2,3])
            #mixed_loss, losses, ygt, ypred = model.compute_loss(batch, mode)
            mixed_loss, losses = model.compute_loss(batch, mode, epoch)
            #gt.extend(ygt)
            #pred.extend(ypred)
            
            for key in dict_loss.keys():
                dict_loss[key] += losses[key]
            
            if mode == "train":
                # backward pass
                mixed_loss.backward()
                # update the weights
                optimizer.step()
            
            batches.append({key: value.detach().cpu() for key, value in batch.items()})
            #batches.append(batch)
     
        #pred = np.stack([t.detach().cpu().numpy() for t in pred])
        #gt = np.stack([t.detach().cpu().numpy() for t in gt])

        #acc = np.sum(gt == pred) / len(gt)
    
    return dict_loss, batches


def train(model, optimizer, iterator, device, epoch, para):
    return train_or_test(model, optimizer, iterator, device, epoch, para, mode="train")


def test(model, optimizer, iterator, device, epoch, para):
    return train_or_test(model, optimizer, iterator, device, epoch, para, mode="test")
