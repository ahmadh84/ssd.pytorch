import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        def print_vals(cy, h, cx, w, fd):
            cy = 0 if cy < 0 else cy
            cy = 1 if cy > 1 else cy
            h = 0 if h < 0 else h
            h = 1 if h > 1 else h
            cx = 0 if cx < 0 else cx
            cx = 1 if cx > 1 else cx
            w = 0 if w < 0 else w
            w = 1 if w > 1 else w
            fd.write("%.5f %.5f %.5f %.5f\n" % (cy-h/2, cy+h/2, cx-w/2, cx+w/2))

        mean = []
        # TODO merge these
        if self.version == 'v2':
            fd = open("anchor_list.txt", "w+")
            for k, f in enumerate(self.feature_maps):
                f_k = self.image_size / self.steps[k]
                print(type(self.steps[k]), type(self.image_size), type(f_k))
                s_k = self.min_sizes[k]/self.image_size
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                print("sz [%d, %d], step %d, f %d, f_k %.2f, s_k %.2f, s_k* %.2f" % (self.min_sizes[k], self.max_sizes[k], self.steps[k], f, f_k, s_k, s_k_prime))

                fd.write("Feature map %d [%dx%d]\n" % (k+1, f, f))

                for i, j in product(range(f), repeat=2):
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    mean += [cx, cy, s_k, s_k]
                    print_vals(mean[-3], mean[-1], mean[-4], mean[-2], fd)
                    if (i,j) == (0,0):
                        print("%d:  %.5f  %.5f" % (k, mean[-2], mean[-1]))

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    mean += [cx, cy, s_k_prime, s_k_prime]
                    print_vals(mean[-3], mean[-1], mean[-4], mean[-2], fd)
                    if (i,j) == (0,0):
                        print("%d:  %.5f  %.5f" % (k, mean[-2], mean[-1]))

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        if (i,j) == (0,0):
                            print("%d:  %.5f  %.5f" % (k, mean[-2], mean[-1]))
                        print_vals(mean[-3], mean[-1], mean[-4], mean[-2], fd)
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                        if (i,j) == (0,0):
                            print("%d:  %.5f  %.5f" % (k, mean[-2], mean[-1]))
                        print_vals(mean[-3], mean[-1], mean[-4], mean[-2], fd)
            fd.close()
        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
