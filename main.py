import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from data_loader import DataSetWrapper
from model import SimCLR

def main(args):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    ### Hyperparameters setting ###
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.temperature
    proj_dim = args.out_dim

    ### DataLoader ###
    dataset = DataSetWrapper(args.batch_size, args.num_worker , args.valid_size, input_shape = (96, 96, 3), strength=args.strength)
    train_loader , valid_loader = dataset.get_data_loaders()

    model = SimCLR(out_dim=proj_dim).to(device)

    ### You may use below optimizer & scheduler ###
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''
    val_loss = 1e9

    def NTXent(z, N):
        z = F.normalize(z, dim=1)
        mask = torch.eye(N*2).to(device).bool()
        tmp = torch.mm(z, z.T).masked_fill(mask, float('-inf'))
        loss_matrix = - F.log_softmax(tmp / T, dim=1) 
        loss = sum(torch.diag(loss_matrix[:N, N:]))+sum(torch.diag(loss_matrix[N:, :N]))
        loss /= 2 * N
        return loss

    def train(xi, xj):
        xi, xj = xi.to(device), xj.to(device)
        h, z = model(xi, xj)
        optimizer.zero_grad()
        loss = NTXent(z, xi.shape[0])
        loss.backward()
        optimizer.step()
        return loss

    def valid(xi, xj):
        xi, xj = xi.to(device), xj.to(device)
        h, z = model(xi, xj)
        loss = NTXent(z, xi.shape[0])
        return loss

    for epoch in range(epochs):
        start = time.time()
        for (xi, xj), _ in train_loader:
            loss = train(xi, xj)
        print("[Epoch %d] Train Loss %f. Time takes %s" %(epoch, loss, time.time()-start)) 

        with torch.no_grad():
            losses = []
            for (val_xi, val_xj), _ in valid_loader:
                losses.append(valid(val_xi, val_xj))
            loss = sum(losses) / len(losses)
            print("[Epoch %d] Valid Loss %f" %(epoch, loss))
            
            if (loss < val_loss):
                val_loss = loss
            else:
                # You have to save the model using early stopping
                torch.save({"model": model.state_dict(), "epoch": epoch, "loss": loss}, "model%f.ckpt" %(args.strength))
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "SimCLR implementation")

    parser.add_argument(
        '--epochs',
        type=int,
        default=40)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5)

    parser.add_argument(
        '--strength',
        type=float,
        default=1.0)

    parser.add_argument(
        '--out_dim',
        type=int,
        default=256)

    parser.add_argument(
        '--num_worker',
        type=int,
        default=8)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)


    args = parser.parse_args()
    main(args)

