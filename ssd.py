import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os
import numpy as np

# weight mappings to VOD code
vod_vgg_t = {0: (1,), \
             2: (3,), \
             5: (6,), \
             7: (8,), \
             10: (11,), \
             12: (13,), \
             14: (15,), \
             17: (18,), \
             19: (20,), \
             21: (22,), \
             24: (24,2,2), \
             26: (24,2,4), \
             28: (24,2,6), \
             31: (24,2,9), \
             33: (24,2,11)}
vod_extras_t = {0: (24,2,13,2,1), \
                1: (24,2,13,2,3), \
                2: (24,2,13,2,5,2,1), \
                3: (24,2,13,2,5,2,3), \
                4: (24,2,13,2,5,2,5,2,1), \
                5: (24,2,13,2,5,2,5,2,3), \
                6: (24,2,13,2,5,2,5,2,5,2,1), \
                7: (24,2,13,2,5,2,5,2,5,2,3)}
vod_pred_t = {0: (24,1,2), \
              1: (24,2,13,1), \
              2: (24,2,13,2,5,1), \
              3: (24,2,13,2,5,2,5,1),
              4: (24,2,13,2,5,2,5,2,5,1), \
              5: (24,2,13,2,5,2,5,2,5,2,5)}

def copy_from_file(trch_var, filepath):
    np_vec = np.fromfile(filepath, dtype=np.float32)
    assert np_vec.size == trch_var.data.numel(), "Not the same number of elements: " + filepath
    tnsr = torch.from_numpy(np_vec)
    trch_var.data.copy_(tnsr)

def write_to_file(trch_var, filepath):
    tnsr = trch_var.cpu().data.numpy()
    tnsr.tofile(filepath)


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # for i,(l,c) in enumerate(zip(loc, conf)):
        #     print(i, l.size())
        #     print(i, c.size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

            # def grad_manip_lclz(grad_x):
            #     tmp = grad_x.data.cpu()
            #     torch.save(tmp, "loc_tnsr.pt")
            #     print(type(grad_x))
            # def grad_manip_clss(grad_x):
            #     tmp = grad_x.data.cpu()
            #     torch.save(tmp, "clss_tnsr.pt")
            #     print(type(grad_x))
            # output[0].register_hook(grad_manip_lclz)
            # output[1].register_hook(grad_manip_clss)

        return output

    def __op_base_net_weights(self, dirpath, weight_file_op=write_to_file):
        for i in range(len(self.vgg)):
            # print(i, type(self.vgg[i]))
            if isinstance(self.vgg[i], torch.nn.modules.conv.Conv2d):
                # print(vod_vgg_t[i])
                fp = '_'.join([str(vt) for vt in vod_vgg_t[i]])
                fp = os.path.join(dirpath, fp)
                weight_file_op(self.vgg[i].weight, fp + "_wgts.data")
                weight_file_op(self.vgg[i].bias, fp + "_bias.data")
        # store weights for L2 norm
        weight_file_op(self.L2Norm.weight, os.path.join(dirpath, "24_1_1_3_wgts.data"))
        # store weights for extras
        for i in range(len(self.extras)):
            # print(i, type(self.extras[i]))
            fp = '_'.join([str(vt) for vt in vod_extras_t[i]])
            fp = os.path.join(dirpath, fp)
            weight_file_op(self.extras[i].weight, fp + "_wgts.data")
            weight_file_op(self.extras[i].bias, fp + "_bias.data")


    def load_weights_from_dir(self, dirpath):
        print(">> LOADING WEIGHTS FROM DIRECTORY: " + dirpath)
        assert os.path.isdir(dirpath), "Directory '" + dirpath + "' doesnt exist"
        self.__op_base_net_weights(dirpath, copy_from_file)
        # load prediction weights
        for i in range(len(self.loc)):
            loc_w = self.loc[i].weight.data
            loc_b = self.loc[i].bias.data
            conf_w = self.conf[i].weight.data
            conf_b = self.conf[i].bias.data

            fp = '_'.join([str(vt) for vt in vod_pred_t[i]])
            fp = os.path.join(dirpath, fp)
            src_w = np.fromfile(fp + "_wgts.data", dtype=np.float32)
            src_b = np.fromfile(fp + "_bias.data", dtype=np.float32)
            src_w = src_w.reshape((loc_w.shape[0]+conf_w.shape[0], loc_w.shape[1], loc_w.shape[2], loc_w.shape[3]))
            src_b = src_b.reshape((loc_w.shape[0]+conf_w.shape[0],))

            n_anchs = int(loc_w.shape[0] / 4)
            loc_i = 0
            clsf_i = 0
            src_i = 0
            for _ in range(n_anchs):
                loc_end = loc_i + 4
                clsf_end = clsf_i + self.num_classes

                loc_w[loc_i+1].copy_(torch.from_numpy(src_w[src_i]))
                loc_w[loc_i+3].copy_(torch.from_numpy(src_w[src_i+1]))
                loc_w[loc_i+0].copy_(torch.from_numpy(src_w[src_i+2]))
                loc_w[loc_i+2].copy_(torch.from_numpy(src_w[src_i+3]))
                loc_b[loc_i+1] = float(src_b[src_i])
                loc_b[loc_i+3] = float(src_b[src_i+1])
                loc_b[loc_i+0] = float(src_b[src_i+2])
                loc_b[loc_i+2] = float(src_b[src_i+3])
                src_i += 4
                conf_w[clsf_i:clsf_end].copy_(torch.from_numpy(src_w[src_i:src_i+self.num_classes]))
                conf_b[clsf_i:clsf_end].copy_(torch.from_numpy(src_b[src_i:src_i+self.num_classes]))

                src_i += self.num_classes

                loc_i = loc_end
                clsf_i = clsf_end

            assert loc_i == loc_w.shape[0]
            assert clsf_i == conf_w.shape[0]
            assert src_i == loc_w.shape[0]+conf_w.shape[0]

    def save_weights_to_dir(self, dirpath):
        print(">> SAVING WEIGHTS TO DIRECTORY: " + dirpath)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        self.__op_base_net_weights(dirpath, write_to_file)
        # store prediction weights
        for i in range(len(self.loc)):
            loc_w = self.loc[i].weight.cpu().data.numpy()
            loc_b = self.loc[i].bias.cpu().data.numpy()
            conf_w = self.conf[i].weight.cpu().data.numpy()
            conf_b = self.conf[i].bias.cpu().data.numpy()

            dest_w = np.empty((loc_w.shape[0]+conf_w.shape[0], loc_w.shape[1], loc_w.shape[2], loc_w.shape[3]), dtype=loc_w.dtype)
            dest_b = np.empty((loc_w.shape[0]+conf_w.shape[0],), dtype=loc_w.dtype)

            n_anchs = int(loc_w.shape[0] / 4)
            loc_i = 0
            clsf_i = 0
            dest_i = 0
            for _ in range(n_anchs):
                loc_end = loc_i + 4
                clsf_end = clsf_i + self.num_classes

                c_loc_w = loc_w[loc_i:loc_end]
                c_loc_b = loc_b[loc_i:loc_end]
                c_conf_w = conf_w[clsf_i:clsf_end]
                c_conf_b = conf_b[clsf_i:clsf_end]

                dest_w[dest_i:dest_i+4] = c_loc_w[[1,3,0,2]]
                dest_b[dest_i:dest_i+4] = c_loc_b[[1,3,0,2]]
                # print(np.all(dest_w[3,:,:,:] == c_loc_w[2,:,:,:]))
                dest_i += 4
                dest_w[dest_i:dest_i+self.num_classes] = c_conf_w
                dest_b[dest_i:dest_i+self.num_classes] = c_conf_b
                dest_i += self.num_classes

                loc_i = loc_end
                clsf_i = clsf_end

            assert loc_i == loc_w.shape[0]
            assert clsf_i == conf_w.shape[0]
            assert dest_i == loc_w.shape[0]+conf_w.shape[0]

            fp = '_'.join([str(vt) for vt in vod_pred_t[i]])
            dest_w.tofile(os.path.join(dirpath, fp + "_wgts.data"))
            dest_b.tofile(os.path.join(dirpath, fp + "_bias.data"))

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return

    return SSD(phase, *multibox(vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
