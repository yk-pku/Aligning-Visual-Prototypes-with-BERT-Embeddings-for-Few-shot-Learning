import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .meta_template import MetaTemplate
from .text_mapping import Text_mapping
from .protonet import ProtoNet
import csv

class ProtoNet_cca(nn.Module):
    def __init__(self, model_func, n_way, n_support, final_dimension, text_vector_dimension,
                visual_net_parameters, matrix_v_path, matrix_t_path, au_change = False,
                initial = True, lam = 0, result_way = 1, loss_lam = 1, full_layer = False):
        super(ProtoNet_cca, self).__init__()
        self.visual_net = ProtoNet(model_func, n_way, n_support)
        # self.text_net = Text_mapping(text_vector_dimension, output_dimension, n_way, n_support, cosine)
        self.text_vector_dimension = text_vector_dimension
        self.final_dimension = final_dimension
        self.matrix_v_path = matrix_v_path
        self.matrix_t_path = matrix_t_path
        self.au_change = au_change
        self.n_way = n_way
        self.n_support = n_support
        self.initial = initial
        self.visual_net_parameters = visual_net_parameters
        # self.text_net_parameters = text_net_parameters
        self.lam = lam
        self.loss_lam = loss_lam
        self.result_way = result_way
        self.full_layer = full_layer

        self.loss_fn = nn.CrossEntropyLoss()

        if self.initial:
            print("initial parameters...")
            vis_tmp = torch.load(self.visual_net_parameters)
            # text_tmp = torch.load(self.text_net_parameters)
            self.visual_net.load_state_dict(vis_tmp['state_dict'])
            # self.text_net.load_state_dict(text_tmp['state_dict'])
            print("initial parameters done")
        print("get matrix...")
        self.matrix_v = self.get_matrix(self.matrix_v_path)
        self.matrix_t = self.get_matrix(self.matrix_t_path)
        print("done")


    def get_matrix(self, path):
        matrix = []
        with open(path, 'r') as f:
            count_ = 0
            reader = csv.reader(f)
            for i in reader:
                # if count_ == 0:
                #     count_ += 1
                #     continue
                line = map(float, i)
                line = list(line)
                matrix.append(line)
        if self.au_change:
            matrix = torch.tensor(matrix, requires_grad = True)
        else:
            matrix = torch.tensor(matrix, requires_grad = False)
        return matrix

    def set_forward(self, x, text_vector, is_feature = False):
        z_support, z_query  = self.visual_net.parse_feature(x, is_feature)
        text_vector = text_vector.cuda()
        self.matrix_t = self.matrix_t.cuda()
        self.matrix_v = self.matrix_v.cuda()
        # text_vector : [5, 21, 300]
        # text_vector = text_vector.contiguous().view(self.n_way*(self.n_support + self.n_query), self.text_vector_dimension)

        z_proto = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #(5, 512)
        z_support_text_vector = text_vector[:, 0, :]
        z_support_text_vector = z_support_text_vector.contiguous().view(-1, self.text_vector_dimension)
        # text_feature = self.text_net.get_text_feature(z_support_text_vector) # (5, 512)
        text_feature = z_support_text_vector @ self.matrix_t
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        dists = euclidean_dist(z_query, z_proto)
        z_query_low = z_query @ self.matrix_v

        if self.full_layer:
            text_feature = self.transfer(text_feature)
            z_query_low = self.transfer(z_query_low)
        cosine_dists = cosine_dist(z_query_low, text_feature)

        scores = -dists + self.loss_lam * cosine_dists


        return scores


    def set_forward_loss(self, x, text_vector):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x, text_vector)

        return self.loss_fn(scores, y_query)

    def train_loop(self, epoch, train_loader, optimizer, logger, logger_file):
        self.train()
        print_freq = 10

        avg_loss=0
        for i, (x, text_vector, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, text_vector)
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
        for i, (x, text_vector, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "protonet do not support way change"
            correct_this, count_this = self.correct(x, text_vector)
            acc_all.append(correct_this/count_this *100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)

        logger_line = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num))
        if logger_file is not None:
            logger_file.write(logger_line + '\n')
        print(logger_line)
        # print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def correct(self, x, text_vector):
        scores = self.set_forward(x, text_vector)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0]==y_query)
        return float(top1_correct), len(y_query)


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

def cosine_dist(x, y):
    x = F.normalize(x, p = 2, dim = 1, eps=1e-12)
    y = F.normalize(y, p = 2, dim = 1, eps=1e-12)
    # x: N * D
    # y: M * D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    yT = torch.transpose(y, 1, 0)

    output = x @ yT

    return output
