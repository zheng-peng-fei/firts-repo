#python data.py cifar10 --aux_out_dataset tinyimages_300k --test_out_dataset tinyimages_300k
# -*- coding: utf-8 -*-
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets import *
from models.wrn_ssnd import *

import wandb

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

'''
This code implements training and testing functions. 
'''


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float,
                    default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int,
                    default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int,
                    default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float,
                    default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability')
# Checkpoints
parser.add_argument('--results_dir', type=str,
                    default='results', help='Folder to save .pkl results.')
parser.add_argument('--checkpoints_dir', type=str,
                    default='checkpoints', help='Folder to save .pt checkpoints.')

parser.add_argument('--load_pretrained', type=str,
                    default='snapshots/pretrained', help='Load pretrained model to test or resume training.')
parser.add_argument('--test', '-t', action='store_true',
                    help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1, help='Which GPU to run on.')
parser.add_argument('--prefetch', type=int, default=4,
                    help='Pre-fetching threads.')
# EG specific
parser.add_argument('--score', type=str, default='SSND', help='SSND|OE|energy|VOS')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--classification', type=boolean_string, default=True)

# dataset related
parser.add_argument('--aux_out_dataset', type=str, default='svhn', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k'],
                    help='Auxiliary out of distribution dataset')
parser.add_argument('--test_out_dataset', type=str,choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k'],
                    default='svhn', help='Test out of distribution dataset')
parser.add_argument('--pi', type=float, default=1,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')

###woods/woods_nn specific
parser.add_argument('--in_constraint_weight', type=float, default=1,
                    help='weight for in-distribution penalty in loss function')
parser.add_argument('--out_constraint_weight', type=float, default=1,
                    help='weight for out-of-distribution penalty in loss function')
parser.add_argument('--ce_constraint_weight', type=float, default=1,
                    help='weight for classification penalty in loss function')
parser.add_argument('--false_alarm_cutoff', type=float,
                    default=0.05, help='false alarm cutoff')

parser.add_argument('--lr_lam', type=float, default=1, help='learning rate for the updating lam (SSND_alm)')
parser.add_argument('--ce_tol', type=float,
                    default=2, help='tolerance for the loss constraint')

parser.add_argument('--penalty_mult', type=float,
                    default=1.5, help='multiplicative factor for penalty method')

parser.add_argument('--constraint_tol', type=float,
                    default=0, help='tolerance for considering constraint violated')

# Energy Method Specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-5.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')  # T = 1 suggested by energy paper

#energy vos method
parser.add_argument('--energy_vos_lambda', type=float, default=2, help='energy vos weight')

# OE specific
parser.add_argument('--oe_lambda', type=float, default=.5, help='OE weight')


# parse argument
args = parser.parse_args()

# method_data_name gives path to the model
if args.score in ['woods_nn']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                               str(args.in_constraint_weight),
                                               str(args.out_constraint_weight),
                                               str(args.ce_constraint_weight),
                                               str(args.false_alarm_cutoff),
                                               str(args.lr_lam),
                                               str(args.penalty_mult),
                                               str(args.pi))
elif args.score == "energy":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                      str(args.m_in),
                                      str(args.m_out),
                                      args.pi)
elif args.score == "OE":
    method_data_name = "{}_{}_{}".format(args.score,
                                   str(args.oe_lambda),
                                   str(args.pi))
elif args.score == "energy_vos":
    method_data_name = "{}_{}_{}".format(args.score,
                                   str(args.energy_vos_lambda),
                                   str(args.pi))
elif args.score in ['woods']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                            str(args.in_constraint_weight),
                                            str(args.out_constraint_weight),
                                            str(args.false_alarm_cutoff),
                                            str(args.ce_constraint_weight),
                                            str(args.lr_lam),
                                            str(args.penalty_mult),
                                            str(args.pi))


state = {k: v for k, v in args._get_kwargs()}
print(state)
from PIL import Image

def recoverdata(data,filename):
    import os

    if not os.path.exists(filename):
        os.makedirs(filename)
    img = data.clone().detach().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename+'/'+str(i)+'.png')

train_loader_in, train_loader_aux_in, train_loader_aux_out, test_loader, test_loader_ood, valid_loader_in, valid_loader_aux_in, valid_loader_aux_out = make_datasets(
    args.dataset, args.aux_out_dataset, args.test_out_dataset, args.pi, state)

loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_out)
first_element = next(iter(train_loader_in))
#print(first_element[1].shape)#长度为2，第一个元素为图片数据、第二个元素为图片标签。形状分别为：torch.Size([128, 3, 32, 32])和torch.Size([128])
data=first_element[0][0]
#print(data.shape)



mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

#原来的图片
dataset=first_element[0]
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture0')

#现在的图片：
r=0.1
tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
tensor_norm=tensor.norm(dim=3).norm(dim=2).norm(dim=1)
for i in range(128):
    tensor[i]=tensor[i]/tensor_norm[i]*r
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture1')

#
r=1
tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
tensor_norm=tensor.norm(dim=3).norm(dim=2).norm(dim=1)
for i in range(128):
    tensor[i]=tensor[i]/tensor_norm[i]*r
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture2')

#
r=10
tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
tensor_norm=tensor.norm(dim=3).norm(dim=2).norm(dim=1)
for i in range(128):
    tensor[i]=tensor[i]/tensor_norm[i]*r
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture3')

#
r=100
tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
tensor_norm=tensor.norm(dim=3).norm(dim=2).norm(dim=1)
for i in range(128):
    tensor[i]=tensor[i]/tensor_norm[i]*r
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture4')

#
r=1000
tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
tensor_norm=tensor.norm(dim=3).norm(dim=2).norm(dim=1)
for i in range(128):
    tensor[i]=tensor[i]/tensor_norm[i]*r
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture5')

#

tensor=torch.FloatTensor(128,3,32,32).uniform_(-2,2)
dataset=first_element[0]+tensor
for i in range(128):
    data=dataset[i]
    recoverdata(data,'Picture6')


for i in range(7):
    image_width=32
    image_height=32
    
    canvas=(Image.new('RGB', (image_width * 10, image_height * 2)))
    # 循环读取并拼接所有图片
    for j in range(20):
        # 打开当前图片
        image = Image.open('Picture'+str(i)+'/'+str(j)+'.png')
        
        # 计算其在拼接图中的位置
        x = (j % 10) * image_width
        y = (j // 10) * image_height
        
        # 将当前图片粘贴到拼接图中对应的位置
        canvas.paste(image, (x, y))
        
        # 保存拼接好的图片
    if not os.path.exists('Picture'):
        os.makedirs('Picture')
    canvas.save('Picture/merged_image'+str(i)+'.png')