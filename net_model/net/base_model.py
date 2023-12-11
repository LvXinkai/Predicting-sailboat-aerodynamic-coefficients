import os
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.opt = None
        self.gpu_ids = None
        self.save_dir = None
        self.device = None
        self.model_names = None
        self.input = None

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.model_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_names = []

    def set_input(self, input_data):
        pass

    # def test(self):
    #     with torch.no_grad():
    #         self.forward()

    # save models to the disk
    def save_networks(self, which_epoch, loss):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s_loss_%s.pth' % (which_epoch, name, str(loss))
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                    # net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, load_path, name='net'):
        net_a = getattr(self, name)
        # print("neta",net_a)
        # net_a = net_a.module
        state_dict_a = torch.load(load_path, map_location=torch.device('cpu'))
        for key in list(state_dict_a.keys()):
            self.__patch_instance_norm_state_dict(state_dict_a, net_a, key.split('.'))
        net_a.load_state_dict(state_dict_a)

    # print network information
    def print_networks(self, verbose=True):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            # print(name)
            # print(self.model_names)
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                # if verbose:
                #     print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('GroupNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

