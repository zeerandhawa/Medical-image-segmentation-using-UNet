import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import UNet
from dataloader import DataLoader


def train_net(net,
              epochs=1,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.0008,
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    loader = DataLoader(data_dir)

    if(torch.cuda.is_available()):
        print('available')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


    net = net.cuda()
    N_train = loader.n_train()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.005)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            optimizer.zero_grad()
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)

            shape = img.shape
            img_batch = torch.from_numpy(img.reshape(1, 1, shape[0], shape[1])).float()
            shape1 = label.shape
            #label = torch.from_numpy(label.reshape(1, 1, shape1[0], shape1[1]))


            #label = torch.from_numpy(label.reshape( 1, shape1[0], shape1[1])).float()
            label = torch.from_numpy(label).float()
            img_batch = Variable(img_batch)
            label = Variable(label)

            # todo: load image tensor to gpu
            if gpu:
                img_batch = img_batch.cuda()
                label = label.cuda()

            # todo: get prediction and getLoss()
            predicted = net.forward(img_batch)


            #loss = criterion(predicted, label)
            loss = getLoss(predicted, label)

            epoch_loss += loss.item()

            print('Training sample %d / %d - Loss: %.6f' % (i + 1, N_train, loss.item()))

            # optimize weights
            loss.backward()
            optimizer.step()

        if(epoch%3 == 0):
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))

        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / i))

    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        for _, (img, label) in enumerate(loader):

            shape = img.shape
            img_torch = torch.from_numpy(img.reshape(1, 1, shape[0], shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()

            img_torch = img_torch.cuda()
            pred = net.forward(img_torch)

            pr = torch.nn.Softmax(dim = 2)

            inb = pr(pred)

            pr_sft = softmax(pred)
            __, pred_soft = torch.max(pr_sft, 1)
            #max_val, pred_soft = pr_sft.max(dim = 1)


            size = pr_sft.size()
            pred_soft = torch.zeros([1,388,388])

            count = 1
            for x in range(size[2]):
                for y in range(size[3]):
                    pr1 = pr_sft[0,0,x,y]
                    #print(pr1)
                    pr2 = pr_sft[0,1,x,y]
                    #print(pr2)
                    if(pr_sft[0,0,x,y]>pr_sft[0,1,x,y]):
                        #print('true')
                        pred_soft[0,x,y] = 1
                        count = count+1
                    #else:
                        #print('false')
            #print("S+str(torch.sum(pred_soft)))


            print('count'+str(count))
            print(pred_soft)
            print(type(pred_soft))
            #print(max_val)
            print(pred_soft.size())
            _, pred_label_1 = torch.max(inb, 1)

            plt.subplot(1, 3, 1)
            plt.imshow(img * 255.)
            #plt.subplot(1, 3, 2)
            #plt.imshow((label - 1) * 255.)
            plt.subplot(1, 3, 2)
            plt.imshow(label-1* 255.)

            plt.subplot(1, 3, 3)
            print('inb dimension :' + str(inb.size()))
            plt.imshow(pred_label_1.cpu().detach().numpy().squeeze() * 255.)
            plt.show()



def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)


def softmax(input):
    # todo: implement softmax function

    # convert to FloatTensor if needed

    exp = torch.exp(input)
    sum = exp[:,0,:,:] + exp[:,1,:,:]

    exp1 = exp.clone()
    exp1[:,0,:,:] = torch.div(exp[:,0,:,:], sum)
    exp1[:,1,:,:] = torch.div(exp[:,1,:,:], sum)
    p = exp1

    return p


def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function

    pred = choose(input, targets)

    print('pred in ce')

    pred = -torch.log(pred)
    print(pred.size())
    #float(1/)
    sz = pred.size()
    ce = torch.mean(pred)
    sum = torch.sum(pred)
    #ce = (1.0/(sz[0]*sz[1]))*sum
    return ce


# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2] * size[3], 3], dtype=int)
    i = 0

    for x in range(size[2]):
        for y in range(size[3]):
            ind[i, :] = [true_labels[x, y], x, y]
            i += 1

    pred = pred_label[0, ind[:, 0], ind[:, 1], ind[:, 2]].view(size[2], size[3])

    return pred


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    #PATH TO SAVED MODEL
    #net.load_state_dict(torch.load('/home/skrandha/sfuhome/UNetFW/data/cells/good_model/CP5.pth'))
    train_net(net=net,
              epochs=args.epochs,
              n_classes=args.n_classes,
              gpu=args.gpu,
              data_dir=args.data_dir)

