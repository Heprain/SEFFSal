import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--inWidth', type=int, default=352, help='Width of RGB image') 
parser.add_argument('--inHeight', type=int, default=352, help='Height of RGB image') 
parser.add_argument('--max_epochs', type=int, default=60, help='Max. number of epochs') 
parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads') 
parser.add_argument('--batch_size', type=int, default=10, help='Batch size') 
parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs') 
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate') 
parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
parser.add_argument('--savedir', default='./results', help='Directory to save the results') 
parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Run on CPU or GPU. If TRUE, then GPU')
parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
parser.add_argument('--depth', type=int, default=1, help='use RGB-D data, default True')
parser.add_argument('--save_depth', default=0, type=int)
parser.add_argument('--depth_weight', type=float, default=0.3, help='idr loss weight, default 0.3') 
parser.add_argument('--criterion', type=str, default='API', help='API or bce') 
parser.add_argument('--pretrained', default=None, help='Pretrained model')

parser.add_argument('--rgb_root', type=str, default='../Dataset/RGB-T/Train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='../Dataset/RGB-T/Train/T/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../Dataset/RGB-T/Train/GT/', help='the training gt images root')
parser.add_argument('--val_rgb_root', type=str, default='../Dataset/RGB-T/Val/RGB/', help='the test rgb images root')
parser.add_argument('--val_depth_root', type=str, default='../Dataset/RGB-T/Val/T/', help='the test depth images root')
parser.add_argument('--val_gt_root', type=str, default='../Dataset/RGB-T/Val/GT/', help='the test gt images root')
parser.add_argument('--test_path',type=str,default='../Dataset/RGB-D/Test/',help='test dataset path')


parser.add_argument('--rgbt', default=1)

parser.add_argument('--net_size', type=str, default='m', help='The size of checkpoints(m/s/t)')

args = parser.parse_args()
