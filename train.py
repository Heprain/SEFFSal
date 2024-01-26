import sys
import os
sys.path.insert(0, '.')

import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

from models import SEFFSal as net

from utils.data import get_loader
from utils.SalEval import SalEval
from utils.criteria import SSIM
from utils.losses import Criterion

from options import args
import time
import numpy as np



def build_ssim_loss(window_size=11):
    return SSIM(window_size=window_size)

@torch.no_grad()
def val(args, val_loader, model, criterion):
    model.eval()
    
    salEvalVal = SalEval()

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):
        if args.depth:
            input, target, depth = batched_inputs
        else:
            input, target = batched_inputs
        start_time = time.time()

        if args.gpu:
            input = input.cuda()
            target = target.cuda()
            if args.depth:
                depth = depth.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        if args.depth:
            depth_var = torch.autograd.Variable(depth).float()
        else:
            depth_var = None

        # run the mdoel
        output1,output2,output3 = model(input_var, depth_var)  
        loss1 = criterion(output1, target_var)
        loss2 = criterion(output2, target_var)
        loss3 = criterion(output3, target_var)

        loss = loss1 + loss2 + loss3

        #torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())


        salEvalVal.addBatch(output1[:,0,:,:], target_var)
        if iter % 5 == 0:
            print('\r[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken), end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = salEvalVal.getMetric()

    return average_epoch_loss_val, F_beta, MAE


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    salEvalTrain = SalEval()  
    epoch_loss = []
    ssim = build_ssim_loss()  

    for iter, batched_inputs in enumerate(train_loader):
        if args.depth:   
            input, target, depth = batched_inputs  
        else:
            input, target = batched_inputs
        start_time = time.time()

        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.gpu == True:  
            input = input.cuda()
            target = target.cuda()
            if args.depth:
                depth = depth.cuda()
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        if args.depth:
            depth_var = torch.autograd.Variable(depth).float()
        else:
            depth_var = None

        # run the model
        output1, output2, output3 = model(input_var, depth_var)
        #output=saliency_maps, depth_output= depth_pred_s1
        loss1 = criterion(output1, target_var)
        loss2 = criterion(output2, target_var)
        loss3 = criterion(output3, target_var)
        # true_depth = depth_var * 0.5 + 0.5
        # loss_depth1 = args.depth_weight * (1 - ssim(depth_output1, true_depth))  
        # loss_depth2 = args.depth_weight * (1 - ssim(depth_output2, true_depth))
        # loss_depth3 = args.depth_weight * (1 - ssim(depth_output3, true_depth))

        loss = loss1 + loss2 + loss3 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())  
        time_taken = time.time() - start_time 

        # Computing F-measure and MAE on GPU
        with torch.no_grad():
            salEvalTrain.addBatch(output1[:,0,:,:] , target_var)
        
        if iter % 5 == 0:
            print('\riteration: [%d/%d] lr: %.7f loss: %.3f time:%.3f' % (iter+cur_iter, max_batches*args.max_epochs, lr, loss.data.item(), time_taken), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)  
    F_beta, MAE = salEvalTrain.getMetric()

    return average_epoch_loss_train, F_beta, MAE, lr  

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1): 
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches*args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter+1) / 200 + 0.1 * args.lr # warm_up
    lr *= lr_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(args):
    model = net.SEFFSal(args.net_size)  
    
    args.savedir = args.savedir + '_ep' + str(args.max_epochs) + '/'  
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.gpu:
        model = model.cuda()  
   
    total_params = sum([np.prod(p.size()) for p in model.parameters()]) 

    print('Total network parameters: ' + str(total_params))
    

    #·································································································
    image_root = args.rgb_root
    gt_root = args.gt_root
    depth_root= args.depth_root

    test_image_root= args.test_rgb_root
    test_gt_root= args.test_gt_root
    test_depth_root= args.test_depth_root
    
    train_data = get_loader(image_root, gt_root, depth_root, batchsize=args.batch_size, trainsize=args.inWidth)

    trainLoader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    val_data = get_loader(test_image_root, test_gt_root, test_depth_root, batchsize=args.batch_size, trainsize=args.inWidth)
    
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=10, num_workers=args.num_workers, pin_memory=False)

    #·································································································

    
    max_batches = len(trainLoader)
    
    print('For each epoch, we have {} batches'.format(max_batches))
    
    if args.gpu:
        cudnn.benchmark = True

    start_epoch = 0
    cur_iter = 0

    if args.resume is not None:  
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile  
    #savedir= savedir/_ep_max_epochs/
    
    if os.path.isfile(logFileLoc):  
        logger = open(logFileLoc, 'a') 
    else: 
        logger = open(logFileLoc, 'w')   
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\n%s\t%s\t%s" % ('Epoch', 'F_beta (val)', 'MAE (val)'))
    logger.flush() 
   
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    criteria = Criterion(args)

    #·································································································

    for epoch in range(start_epoch, args.max_epochs):  
        lossTr, F_beta_tr, MAE_tr, lr = \
            train(args, trainLoader, model, criteria, optimizer, epoch, max_batches, cur_iter)  
        cur_iter += len(trainLoader)  
        torch.cuda.empty_cache()

        # evaluate on validation set
        if epoch == 0:
            continue
        
        lossVal, F_beta_val, MAE_val = val(args, valLoader, model, criteria)
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f" % (epoch, F_beta_val, MAE_val))
        logger.flush()

        torch.save({
            'epoch': epoch,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': F_beta_tr,
            'F_val': F_beta_val,
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar') 

        # save the model also
        model_file_name = args.savedir + '/model_' + str(epoch) + '.pth'
        if epoch % 1 == 0 and epoch > args.max_epochs * 0.9: 
            torch.save(model.state_dict(), model_file_name)  

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F_beta(tr) = %.4f\t F_beta(val) = %.4f" \
                % (epoch, lossTr, lossVal, F_beta_tr, F_beta_val))
        torch.cuda.empty_cache()
    logger.close()

if __name__ == '__main__':
    
    # print('Called with args:')
    # print(args)
    main(args)  
