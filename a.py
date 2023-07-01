"""
"""
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets import *
from models.wrn_ssnd import *
import numpy as np
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
train_loader_in, train_loader_aux_in, train_loader_aux_out, test_loader, test_loader_ood, valid_loader_in, valid_loader_aux_in, valid_loader_aux_out = make_datasets(
    args.dataset, args.aux_out_dataset, args.test_out_dataset, args.pi, state)

rng = np.random.default_rng(args.seed)

def mix_batches(aux_in_set, aux_out_set):
    '''
    Args:
        aux_in_set: minibatch from in_distribution
        aux_out_set: minibatch from out distribution

    Returns:
        mixture of minibatches with mixture proportion pi of aux_out_set
    '''

    # create a mask to decide which sample is in the batch
    mask = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi, args.pi])

    aux_out_set_subsampled = aux_out_set[0][mask]
    aux_in_set_subsampled = aux_in_set[0][np.invert(mask)]

    # note: ordering of aux_out_set_subsampled, aux_in_set_subsampled does not matter because you always take the sum
    aux_set = torch.cat((aux_out_set_subsampled, aux_in_set_subsampled), 0)

    return aux_set




loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_out)
u=0
min=2
max=-2
#for in_set, aux_in_set, aux_out_set in loaders:
#    aux_set = mix_batches(aux_in_set, aux_out_set)
    #tensor_1 = torch.randn(128,3,32,32)
    #data = torch.cat((in_set[0], tensor_1), 0)
    #u=u+1

#print(data[3])
#tensor_1 = torch.randn(256,3,32,32)
tensor0=tensor=torch.FloatTensor(1,3,32,32).uniform_(-3, -1)
for i in range(127):
    if i%2==0:
        tensor=torch.FloatTensor(1,3,32,32).uniform_(1, 3)
    else:
        tensor=torch.FloatTensor(1,3,32,32).uniform_(-3, -1)

    tensor0=torch.cat((tensor0,tensor),dim=0)

#更好的方法：生成两个tensor,size为(128,3,32,32),然后从这两个tensor里选组合
#
print(tensor0.shape)
#print(min)
#print(max)
#print(u)
#数据集一般都是：torch.Size([256, 3, 32, 32])   
#bash run.sh woods cifar10 dtd dtd:torch.Size([256, 3, 32, 32])    
#bash run.sh woods cifar100 dtd svhn:torch.Size([256, 3, 32, 32]) 有30个batchsize
#loader里有一个epoch所需要的所有数据,把所有的数据按照batachsize的大小打包
#一个batchsize=128,但是是对in_set[0]和aux_set同时打包
#zip会把多元的元素丢掉

#最大值和最小值分别为：
#tensor(-1.9889)
#tensor(2.1256)
#torch.randn()是用于生成正态随机分布张量的函数，从标准正态分布中随机抽取一个随机数生成一个张量，其调用方法如下所示：
"""
import torch 
import numpy as np
def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(1)
rng = np.random.default_rng(2)
tensor0=torch.FloatTensor(393216).uniform_(0, 0.5)
tensor1=torch.FloatTensor(393216).uniform_(-0.5, 0)
mask = rng.choice(a=[False, True], size=(393216), p=[0.5, 0.5])
tensor0_subsampled = tensor0[mask]
tensor1_subsampled = tensor1[np.invert(mask)]
tensor = torch.cat((tensor0_subsampled, tensor1_subsampled), 0)
tensor=tensor.reshape(128,3,32,32)
print(tensor0[0])
print(tensor1[0])
print(tensor.shape)



import torch 
import numpy as np
def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(1)
rng = np.random.default_rng(2)
tensor0=torch.FloatTensor(8).uniform_(0, 0.5)
tensor1=torch.FloatTensor(8).uniform_(-0.5, 0)
mask = rng.choice(a=[False, True], size=(8), p=[0.5, 0.5])
tensor0_subsampled = tensor0[mask]
tensor1_subsampled = tensor1[np.invert(mask)]
tensor = torch.cat((tensor0_subsampled, tensor1_subsampled), 0)
tensor=tensor.reshape(2,4)

#print(tensor0)
print(tensor.shape[0])
#print(tensor)
"""