"""
author: Min Seok Lee and Wooseok Shin
"""
import torch
import torch.nn.functional as F


def Optimizer(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


def Scheduler(args, optimizer):
    if args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.patience)
    elif args.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
    return scheduler


def Criterion(args):
    if args.criterion == 'API':
        criterion = adaptive_pixel_intensity_loss
    elif args.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    return criterion



def adaptive_pixel_intensity_loss(pred, mask):
    mask = mask[:,0,:,:]
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce0 = F.binary_cross_entropy(pred[:,0,:,:], mask, reduce=None)
    bce1 = F.binary_cross_entropy(pred[:,1,:,:], mask, reduce=None)
    bce2 = F.binary_cross_entropy(pred[:,2,:,:], mask, reduce=None)
    bce3 = F.binary_cross_entropy(pred[:,3,:,:], mask, reduce=None)
    # bce4 = F.binary_cross_entropy(pred[:,4,:,:], mask, reduce=None)
    # bce = (bce0 + bce1 + bce2 + bce3 + bce4)/5
    bce = (bce0 + bce1 + bce2 + bce3)/4

        
    abce = (omega * bce).sum(dim=(1, 2)) / (omega + 0.5).sum(dim=(1, 2))

    inter0 = ((pred[:,0,:,:] * mask) * omega).sum(dim=(1, 2))
    union0 = ((pred[:,0,:,:] + mask) * omega).sum(dim=(1, 2))
    inter1 = ((pred[:,1,:,:] * mask) * omega).sum(dim=(1, 2))
    union1 = ((pred[:,1,:,:] + mask) * omega).sum(dim=(1, 2))
    inter2 = ((pred[:,2,:,:] * mask) * omega).sum(dim=(1, 2))
    union2 = ((pred[:,2,:,:] + mask) * omega).sum(dim=(1, 2))
    inter3 = ((pred[:,3,:,:] * mask) * omega).sum(dim=(1, 2))
    union3 = ((pred[:,3,:,:] + mask) * omega).sum(dim=(1, 2))
    # inter4 = ((pred[:,4,:,:] * mask) * omega).sum(dim=(1, 2))
    # union4 = ((pred[:,4,:,:] + mask) * omega).sum(dim=(1, 2))
    # inter =  (inter0 + inter1 + inter2 + inter3 + inter4) / 5
    # union =  (union0 + union1 + union2 + union3 + union4) / 5
    inter =  (inter0 + inter1 + inter2 + inter3 ) / 4
    union =  (union0 + union1 + union2 + union3 ) / 4

    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae0 = F.l1_loss(pred[:,0,:,:], mask, reduce=None)
    mae1 = F.l1_loss(pred[:,1,:,:], mask, reduce=None)
    mae2 = F.l1_loss(pred[:,2,:,:], mask, reduce=None)
    mae3 = F.l1_loss(pred[:,3,:,:], mask, reduce=None)
    # mae4 = F.l1_loss(pred[:,4,:,:], mask, reduce=None)
    # mae = (mae0 + mae1 + mae2 + mae3 + mae4) / 5
    mae = (mae0 + mae1 + mae2 + mae3) / 4

    amae = (omega * mae).sum(dim=(1, 2)) / (omega - 1).sum(dim=(1, 2))

    #1. return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()
    #2. return (0.7 * abce + 0.7 * amae).mean()
    #3. return (1 * abce + 0.1 * amae).mean()
    #4. return (1 * abce + 0.1 * aiou + 0.5 * amae).mean()
    #5. return (1 * abce + 0.7 * amae).mean()
    return (1 * abce + 0.1 * aiou + 0.1 * amae).mean()




# def adaptive_pixel_intensity_loss(pred, mask):
    # w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    # w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    # w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
# 
    # omega = 1 + 0.5 * (w1 + w2 + w3) * mask
# 
    # bce = F.binary_cross_entropy(pred, mask, reduce=None)
    # abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))
# 
    # inter = ((pred * mask) * omega).sum(dim=(2, 3))
    # union = ((pred + mask) * omega).sum(dim=(2, 3))
    # aiou = 1 - (inter + 1) / (union - inter + 1)
# 
    # mae = F.l1_loss(pred, mask, reduce=None)
    # amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))
# 
    # return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()