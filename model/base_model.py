import os
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import util

class BaseModel(nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.log_dir, opt.experiment_name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.visual_pose_names = []
        self.image_paths = []
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.compile = opt.compile

    def set_input(self, input):
        self.input = input

    # return visualization images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)

                if "heatmap" in name:
                    is_heatmap = True
                else:
                    is_heatmap = False

                if "video" in name:
                    is_video = True
                else:
                    is_video = False

                visual_ret[name] = util.tensor2im(value.data, is_heatmap=is_heatmap, is_video=is_video)

        return visual_ret

    # return visualization 3d data
    def get_current_visuals_pose(self, step=None, save_dir_pose=None):
        visual_pose_ret = OrderedDict()
        for name in self.visual_pose_names:
            if isinstance(name, str):
                value = getattr(self, name)

                if "video" in name:
                    is_video = True
                else:
                    is_video = False

                visual_pose_ret[name] = util.tensor2pose(value.data, is_video=is_video)

        return visual_pose_ret

    # save models
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)

                if self.opt.compile:
                    torch.save(net.state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        net.cuda()


    # load models
    def load_networks(self, which_epoch=None, net=None, path_to_trained_weights=None):

        if which_epoch is not None:
            for name in self.model_names:
                print(name)
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (which_epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_'+name)

                    if self.compile:
                        try:
                            net.load_state_dict(self.fix_model_state_dict(torch.load(save_path), key="_orig_mod."))
                        except:
                            net.load_state_dict(torch.load(save_path))
                    else:
                        net.load_state_dict(torch.load(save_path))

                    print('Loaded trained model: {}'.format(os.path.basename(save_filename)))

                    if not self.isTrain:
                        net.eval()

        elif (net is not None) and (path_to_trained_weights is not None):
            if self.compile:
                try:
                    net.load_state_dict(self.fix_model_state_dict(torch.load(path_to_trained_weights), key="_orig_mod."))
                except:
                    net.load_state_dict(torch.load(path_to_trained_weights))
            else:
                net.load_state_dict(torch.load(path_to_trained_weights))

            print('Loaded trained model: {}'.format(os.path.basename(path_to_trained_weights)))

    def fix_model_state_dict(self, state_dict, key="module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith(key):
                name = name[len(key):]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict