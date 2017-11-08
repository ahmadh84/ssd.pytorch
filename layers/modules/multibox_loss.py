import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp

iteration = 0

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets, fd, fd2):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        # print(loc_data.size())
        # print(conf_data.size())
        # print(priors.size())
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        # print(num, num_priors)
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        obj_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # print(targets[idx].size())
            truths = targets[idx][:, :-1].data
            # print(truths)
            labels = targets[idx][:, -1].data
            # print(labels)
            defaults = priors.data
            overlaps = match(self.threshold, truths, defaults, self.variance, labels,
                             loc_t, conf_t, obj_t, idx)
            # match(self.threshold, truths, defaults, self.variance, labels,
            #       loc_t, conf_t, idx)

            for obj_i in range(truths.size(0)):
                print(str(obj_i+1) + " has " + str(obj_t[idx].eq(obj_i+1).long().sum()) + " matches - " + \
                       ("%.5f, %.5f, %.5f, %.5f" % (truths[obj_i][1], truths[obj_i][3], truths[obj_i][0], truths[obj_i][2])) + " class %d" % labels[obj_i])
            print("Unmatched anchors: ", obj_t[idx].eq(0).long().sum(), "\n")

            if iteration == 4 and (idx == 0 or idx == 8 or idx == 9 or idx == 10 or idx == 11):
                print("\nObject %d" % (idx+1))
                curr_anch_sel = torch.nonzero(conf_t[idx]).squeeze(1)
                obj_sel = obj_t[idx][curr_anch_sel] - 1
                for i in range(curr_anch_sel.size(0)):
                    anch_idx = curr_anch_sel[i]
                    obj_idx = obj_sel[i]
                    print("Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                if idx == 0:
                    anch_idx = 789; obj_idx = 3
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                if idx == 8:
                    anch_idx = 2324; obj_idx = 2
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                    anch_idx = 2348; obj_idx = 3
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                    anch_idx = 2375; obj_idx = 4
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                if idx == 10:
                    anch_idx = 956; obj_idx = 1
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))
                if idx == 11:
                    anch_idx = 3327; obj_idx = 1
                    print("-> Sample %d - anchor %d - matched obj %d - overlap %f" % (idx+1, anch_idx+1, obj_idx+1, overlaps[obj_idx, anch_idx]))

        #     print(torch.nonzero(obj_t[1] == 3))
        #     print(torch.nonzero(obj_t[1] == 6))

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # print(conf_t.size())

        pos = conf_t > 0
        # print(torch.nonzero(pos.data))

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t2 = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t2, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        print(num_neg)

        # print(idx_rank)

        # print(idx_rank.size())
        # print("# +ve samples: ", num_pos.sum().data[0])
        # print(num_pos)
        # print("# -ve samples: ", num_neg.sum().data[0])
        # print(num_neg)
        # print("Labels mat: ", conf_t.size())
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print(loss_c[neg])
        # assert neg[pos].long().sum().data[0] == 0, "Negative positives overlap"
        # assert pos[neg].long().sum().data[0] == 0, "Negative positives overlap"

        torch.save(neg.data.cpu(), 'ngtv_smplng_neg_' + str(iteration) + '.pt')
        torch.save(loss_c.data.cpu(), 'ngtv_smplng_lossc_' + str(iteration) + '.pt')

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # # if you want no negative samples
        # conf_p = conf_data[(pos_idx).gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_t[(pos).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        psv_conf_p = conf_data[pos_idx.gt(0)].view(-1, self.num_classes)
        loss_c_psv = F.cross_entropy(psv_conf_p, conf_t[pos.gt(0)], size_average=False)
        ngv_conf_p = conf_data[neg_idx.gt(0)].view(-1, self.num_classes)
        loss_c_ngv = F.cross_entropy(ngv_conf_p, conf_t[neg.gt(0)], size_average=False)

        print("Localization loss\t %.15f (# %d)" % (loss_l.data[0], pos.gt(0).long().sum().data[0]))
        print("Classification +ve loss\t %.15f (# %d)" % (loss_c_psv.data[0], pos.gt(0).long().sum().data[0]))
        print("Classification -ve loss\t %.15f (# %d)" % (loss_c_ngv.data[0], neg.gt(0).long().sum().data[0]))
        print("Classification loss\t %.15f (# %d)" % (loss_c.data[0], (pos+neg).gt(0).long().sum().data[0]))

        fd.write('%.6f %.6f %.6f %d %d\n' % (loss_l.data[0], loss_c_psv.data[0], loss_c_ngv.data[0], pos.gt(0).long().sum().data[0], neg.gt(0).long().sum().data[0]))
        fd.flush()

        global iteration
        pos_ind = torch.nonzero(pos.data)
        neg_ind = torch.nonzero(neg.data)
        tot_l = 0
        tol_cp = 0
        tol_cn = 0
        # write individual localization and positive classification losses to file
        for i in range(pos_ind.size(0)):
            smpl_ind, anch_ind = pos_ind[i][0], pos_ind[i][1]

            psv_conf_p = conf_data[smpl_ind][anch_ind].view(-1, self.num_classes)
            loss_ce = F.cross_entropy(psv_conf_p, conf_t[smpl_ind][anch_ind], size_average=False)
            fd2.write("%d %d %d %d %.8f\n" % (iteration, 2, smpl_ind+1, anch_ind+1, loss_ce.data[0]))
            tol_cp += loss_ce.data[0]

            loc_p = loc_data[smpl_ind][anch_ind].view(-1, 4)
            loc_t2 = loc_t[smpl_ind][anch_ind].view(-1, 4)
            loss_l2 = F.smooth_l1_loss(loc_p, loc_t2, size_average=False)
            fd2.write("%d %d %d %d %.8f\n" % (iteration, 1, smpl_ind+1, anch_ind+1, loss_l2.data[0]))
            tot_l += loss_l2.data[0]
        # write individual negative classification losses to file
        for i in range(neg_ind.size(0)):
            smpl_ind, anch_ind = neg_ind[i][0], neg_ind[i][1]
            ngv_conf_p = conf_data[smpl_ind][anch_ind].view(-1, self.num_classes)
            l = F.cross_entropy(ngv_conf_p, conf_t[smpl_ind][anch_ind], size_average=False)
            fd2.write("%d %d %d %d %.8f\n" % (iteration, 3, smpl_ind+1, anch_ind+1, l.data[0]))
            tol_cn += l.data[0]

        iteration += 1

        import math
        # print(loss_l.data[0] - tot_l)
        if math.fabs(loss_l.data[0] - tot_l) > 1e-3:
            print("GOOD LORD - lclz loss diff => %f - %f = %f" % (loss_l.data[0], tot_l, loss_l.data[0] - tot_l))
        # print(loss_c_psv.data[0] - tol_cp)
        if math.fabs(loss_c_psv.data[0] - tol_cp) > 1e-2:
            print("GOOD LORD - +ve clsf loss diff => %f - %f = %f" % (loss_c_psv.data[0], tol_cp, loss_c_psv.data[0] - tol_cp))
        # print(loss_c_ngv.data[0] - tol_cn)
        if math.fabs(loss_c_ngv.data[0] - tol_cn) > 1e-2:
            print("GOOD LORD - +ve clsf loss diff => %f - %f = %f" % (loss_c_ngv.data[0], tol_cn, loss_c_ngv.data[0] - tol_cn))

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
