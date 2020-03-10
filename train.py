import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import sys
import time
from utils.util import *
from utils.save import *
from torchvision import datasets, transforms, utils
import torchvision.models as models
import numpy as np
from tqdm import tqdm

def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()
    acc= AverageMeter()
    log = Log()
    
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    # switch to train mode
    model.train()

    for i, (data, target, _) in tqdm(enumerate(train_loader), desc=f'epoch:{epoch:02}', total=len(train_loader)):
        if args.gpu is not None:
            data = data.cuda()
            target = target.cuda()

        out1, out2, out3, _ = model(data)
        out = out1 + out2 + 0.1 * out3

        loss1 = criterion(out1, target)
        loss2 = criterion(out2, target)
        loss3 = criterion(out3, target)
        
        loss = loss1 + loss2 + 0.1 * loss3
        
        # measure accuracy and record loss
        #prec1, prec5 = accuracy(out, target, topk=(1, 5))  # this is metric on trainset
        prec = accuracy_single(out, target)
        batchsize = data.size(0)
        losses.update(loss.item()  , batchsize)

        if np.isnan(losses.val):
            sys.exit('Loss diverged')

        losses1.update(loss1.item(), batchsize)
        losses2.update(loss2.item(), batchsize)
        losses3.update(loss3.item(), batchsize)
        #top1.update(prec1[0], batchsize)
        #top5.update(prec5[0], batchsize)
        acc.update(prec.item())
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % args.print_freq == 0:
            # print('DFL-CNN <==> Train Epoch: [{0}][{1}/{2}]\n'
            #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #     'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
            #     'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
            #     'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\n'
            #     'Accuracy {acurracy.val:.3f} ({acurracy.avg:.3f})'.format(
            #     #epoch, i, len(train_loader), loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, top1=top1, top5=top5))
            #     epoch, i, len(train_loader), loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, acurracy=acc))
            #
            totalloss = [losses, losses1, losses2, losses3]
            log.save_train_info(epoch, i, len(train_loader), totalloss, acc)
        #if i >= 10: break

    print('\nDFL-CNN <==> Train Epoch: [{0}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
          'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
          'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
          'Accuracy {acurracy.val:.3f} ({acurracy.avg:.3f})'.format(
        # epoch, i, len(train_loader), loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, top1=top1, top5=top5))
        epoch,  loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, acurracy=acc))

    return acc.avg


