import sys
sys.path.insert(0, '.')

import torch
import cv2
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from options import args

from models import SEFFSal as net
from tqdm import tqdm



def get_mean_set(args):
    mean = [0.406, 0.456, 0.485] #BGR
    std = [0.225, 0.224, 0.229]
    return mean, std


@torch.no_grad()
def validateModel(args, model, image_list, label_list, depth_list, savedir):
    mean, std = get_mean_set(args)  
    # evaluate = SalEval()

    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])  
        label = cv2.imread(label_list[idx], 0)  
        label = label / 255   
                              
        depth = cv2.imread(depth_list[idx], 0) / 255  

        if args.depth:
            depth -= 0.5
            depth /= 0.5
            depth = cv2.resize(depth, (args.inWidth, args.inHeight))
            depth = torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
            depth_variable = Variable(depth)
        else:
            depth_variable = None

        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.inWidth, args.inHeight)) 
        img = img.astype(np.float32) / 255.  
        img -= mean  
        img /= std

        img = img[:,:, ::-1].copy()  
        
        img = img.transpose((2, 0, 1)) 
        img_tensor = torch.from_numpy(img)  
        img_tensor = torch.unsqueeze(img_tensor, 0)  
        
        img_variable = Variable(img_tensor)


        label = torch.from_numpy(label).float().unsqueeze(0).cuda()

        if args.gpu:
            img_variable = img_variable.cuda()
        

        img_out1, _, _ = model(img_variable, depth=depth_variable)

        img_out1 = F.interpolate(img_out1, size=image.shape[:2], mode='bilinear', align_corners=False)
      

        if args.save_depth:  
            depth_out = F.interpolate(depth_out, size=image.shape[:2], mode='bilinear', align_corners=False)
            depthMap_numpy = (depth_out * 255).data.cpu().numpy()[0, 0].astype(np.uint8)
            depthMapGT_numpy = ((depth_variable[0,0] *0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
      
        
        salMap_numpy = (img_out1*255).data.cpu().numpy()[0,0].astype(np.uint8)


        name = image_list[idx].split('/')[-1]
        cv2.imwrite(osp.join(savedir, name[:-4] + '.png'), salMap_numpy)
        
        if args.save_depth:
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth_pred.png'), depthMap_numpy)
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth.png'), depthMapGT_numpy)

    # F_beta, MAE = evaluate.getMetric()
    # print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))

def main(args, data_list):
    # read all the images in the folder
    image_list = list()  
    label_list = list()  
    depth_list = list()

    image_root = args.test_path + data_list + '/RGB/'  
    gt_root = args.test_path + data_list + '/GT/'  
    dep_root = args.test_path + data_list + '/depth/' 

    image_list_name = os.listdir(image_root)
    label_list_name = os.listdir(gt_root)
    depth_list_name = os.listdir(dep_root)

    for name in image_list_name:
        image_list.append(image_root + name)
    for name in label_list_name:
        label_list.append(gt_root + name)
    for name in depth_list_name:
        depth_list.append(dep_root + name)

    model = net.SEFFSal(args.net_size)
    
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    print("loading saliency pretrained model")  
    state_dict = torch.load(args.pretrained)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print(args.pretrained, "does not exactly match params")
    print("loaded saliency pretrained model")

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()
    
    savedir = args.savedir + '/' + data_list + '/'
    if not osp.isdir(savedir):
        os.makedirs(savedir)

    validateModel(args, model, image_list, label_list, depth_list, savedir)


if __name__ == '__main__':

    # print('Called with args:')
    # print(args)
    torch.backends.cudnn.benchmark = True

    data_lists = ['NJU2K','NLPR','STERE','LFSD','SIP']

    for data_list in data_lists:
        print("processing ", data_list)
        main(args, data_list)
