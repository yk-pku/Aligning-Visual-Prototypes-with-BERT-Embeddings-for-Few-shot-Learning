import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

    def train_loop(self, epoch, train_loader, optimizer, logger, logger_file):
        self.train()
        print_freq = 10

        avg_loss=0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.data.item()

            if i % print_freq==0:
                logger_line = 'Epoch {:d}  Batch {:d}/{:d}  Loss {:f}  Lr {:f}'.format(
                            epoch, i, len(train_loader), avg_loss/float(i+1), optimizer.param_groups[0]['lr'])
                logger_file.write(logger_line + '\n')
                print(logger_line)
                # print('Epoch {:d}  Batch {:d}/{:d}  Loss {:f}  Lr {:f}'.format(
                #             epoch, i, len(train_loader), avg_loss/float(i+1), optimizer.param_groups[0]['lr']))
            logger.add_scalar('loss', avg_loss/float(i+1), epoch + i + 1)

    def test_loop(self, test_loader, logger_file = None, return_std=False):
        correct =0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            # import pdb; pdb.set_trace()
            assert self.n_way  ==  x.size(0), "protonet do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/count_this *100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)

        logger_line = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num))
        if logger_file is not None:
            logger_file.write(logger_line + '\n')
        print(logger_line)
        
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
