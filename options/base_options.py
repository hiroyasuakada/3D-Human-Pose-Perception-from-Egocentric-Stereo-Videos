import argparse
import os
from utils import util
import torch
import torch.distributed as dist

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        ### basic define ###
        self.parser.add_argument('--project_name', type=str, default='project_name',
                                help='name of the project. This is used for wnadb')
        self.parser.add_argument('--experiment_name', type=str, default='experiment',
                                help='name of the experiment')
        self.parser.add_argument('--log_dir', type=str, default='log',
                                help='It decides where to store samples and models of the experiment')
        self.parser.add_argument('--save_dir_pose', type=str, default='./results',
                                help = 'saves poses here')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                help='which epoch to load')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='unrealego2_pose_qa_df',
                                help='choose which model to use')
        self.parser.add_argument('--type_local_pose', type=str, default='gt_local_device_pose',
                                help='[gt_local_device_pose|gt_local_pose]')
        self.parser.add_argument('--init_ImageNet', action='store_true',
                                help='If true, use ImageNet initialization for the backbone')
        self.parser.add_argument('--model_name', type=str, default='resnet18',
                                help='name of the backbone')
        self.parser.add_argument('--path_to_trained_heatmap', type=str, default=None,
                                help='path to weights of the trained heatmap estimator')
        self.parser.add_argument('--path_to_trained_pose', type=str, default=None,
                                help='path to weights of the trained pose estimator')
        self.parser.add_argument('--path_to_trained_heatmap_left', type=str, default=None,
                                help='path to weights of the trained heatmap estimator')
        self.parser.add_argument('--path_to_trained_heatmap_right', type=str, default=None,
                                help='path to weights of the trained heatmap estimator')

        ### dataset parameters ###
        self.parser.add_argument('--data_dir', type=str, default="/CT/UnrealEgo/static00/UnrealEgoData_npy",
                                help='training, validation, and testing dataset')
        self.parser.add_argument('--metadata_dir', type=str, default=None,
                                help='training, validation, and testing dataset')
        self.parser.add_argument('--depth_dir_name', type=str, default=None,
                                help='experiment name of sfm depth generation')
        self.parser.add_argument('--seq_len', type=int, default=0,
                                help='number of sequential images to be processd at one iteration')
        self.parser.add_argument('--num_frame_skip', type=int, default=1,
                                help='number of frames to skip for video-based methods')
        self.parser.add_argument('--num_heatmap', type=int, default=15,
                                help='# of heatmaps')
        self.parser.add_argument('--num_threads', default=2, type=int,
                                help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                help='input batch size')
        self.parser.add_argument('--load_size_rgb', nargs='+', type=int, default=[256, 256],
                                help='scale images to this size')
        self.parser.add_argument('--load_size_heatmap', nargs='+', type=int, default=[64, 64],
                                help='scale images to this size')

        ### network structure define ##
        self.parser.add_argument('--init_type', type=str, default='kaiming',
                                help='network initialization [normal|xavier|kaiming]')
        self.parser.add_argument('--ae_hidden_size', type=int, default=20,
                                help='# of channels at the bottom of AutoEncoder')
        self.parser.add_argument("--hidden_dim", type=int, default=256,
                                help="dim of output tensor after linear transformation in transformer")
        self.parser.add_argument("--dim_feedforward", type=int, default=1024,
                                help="dim of output tensor after linear transformation in transformer")
        self.parser.add_argument("--enc_layers", type=int, default=6,
                                help="num of transformer blocks")
        self.parser.add_argument("--dec_layers", type=int, default=6,
                                help="num of transformer blocks")
        self.parser.add_argument("--nheads", type=int, default=8,
                                help="num of heads in transformer")
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                help='dropout rate for transformer')
        self.parser.add_argument('--pre_norm', action='store_true',
                                help='If true, use pre norm')

        ### other settings
        self.parser.add_argument('--use_slurm', action='store_true',
                                help='If true, use slurm cluster')
        self.parser.add_argument('--use_amp', action='store_true',
                                help='Use AMP FP16 training')
        self.parser.add_argument('--init_scale', type=float, default=65536.0,
                                help='init scale for gradscaler in amp')
        self.parser.add_argument('--compile', action='store_true',
                                help='compile network for faster training with Pytorch2.0')
        self.parser.add_argument("--seq_len_test_eval_start", type=int, default=0,
                                help="sequence length used as a criterion for fair evaluation with temporal models")


    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt=self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                self.opt.gpu_ids.append(id)

        return self.opt


def print_opt(opt):
    args = vars(opt)

    print('--------------Options--------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('----------------End----------------')

    # save to the disk
    expr_dir = os.path.join(opt.log_dir, opt.experiment_name)
    util.mkdirs(expr_dir)

    if opt.isTrain:
        file_name = os.path.join(expr_dir, 'train_opt.txt')
    else:
        file_name = os.path.join(expr_dir, 'test_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('--------------Options--------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('----------------End----------------\n')