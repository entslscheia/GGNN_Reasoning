import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Model_plus import GGNN_plus
from data.dataset_plus import ABoxDataset_plus
from utils.train_plus import train
from utils.test_plus import test
from data.dataloader import ABoxDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--annotation_dim', type=int, default=20, help='annotation dimension for nodes')
parser.add_argument('--state_dim', type=int, default=20, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='probability of dropout')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--use_bias', action='store_true', help='enables bias for edges', default=True)
parser.add_argument('--verbal', action='store_true', help='print training info or not', default=True)
parser.add_argument('--manualSeed', type=int, help='manual seed', default=983)

opt = parser.parse_args()
print(opt)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

opt.dataroot = 'data/train.test.json'
fileName = opt.dataroot[5:]

def main(opt):
    train_dataset = ABoxDataset_plus(opt.dataroot, True)
    train_dataloader = ABoxDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=opt.workers)

    test_dataset = ABoxDataset_plus(opt.dataroot, False)
    test_dataloader = ABoxDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=opt.workers)

    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    net = GGNN_plus(train_dataset.n_node, train_dataset.n_edge_types*2, train_dataset.n_types, opt)    # times 2 because it's directed
    net.double()
    # print(net)

    criterion = nn.BCELoss()

    # print(opt.cuda)
    # print(opt.niter)
    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    best_acc = 0.0     # best accuracy has been achieved
    num_of_dec = 0     # number of epochs have a decline of accuracy, used for early stop
    acc_last_iter = 0.0  # accuracy of the last iteration
    for epoch in range(0, opt.niter):
        if num_of_dec >= 5:
            print("Early stop! The accuracy has been dropped for 5 iterations!")
            break
        train(epoch, train_dataloader, train_dataset, net, criterion, optimizer, opt)
        acc = test(test_dataloader, test_dataset, net, criterion, opt)
        if acc > best_acc:
            best_acc = acc
            print("Best accuracy by far: ", best_acc, '%')
            torch.save(net, './' + fileName + str(opt.n_steps) + '_model.pth')
        if acc >= acc_last_iter:
            num_of_dec = 0
        else:
            num_of_dec += 1
        print("The best accuracy achieved by far: ", best_acc, '%')


if __name__ == "__main__":
    main(opt)

