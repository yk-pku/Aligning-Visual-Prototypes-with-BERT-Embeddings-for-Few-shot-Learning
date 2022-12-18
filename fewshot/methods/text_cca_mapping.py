import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .meta_template import MetaTemplate
import csv

class Text_cca_mapping(nn.Module):
    def __init__(self, text_vector_dimension, output_dimension, n_way, n_support,
                matrix_v_path, matrix_t_path,  cosine = False, seperate = False):
        super(Text_cca_mapping, self).__init__()
        self.text_vector_dimension = text_vector_dimension
        self.output_dimension = output_dimension
        self.cosine = cosine
        self.n_way = n_way
        self.n_support = n_support
        self.matrix_v_path = matrix_v_path
        self.matrix_t_path = matrix_t_path
        self.seperate = seperate

        if self.seperate:
            self.mapping_v_model = nn.Linear(self.output_dimension, self.output_dimension)
            self.mapping_t_model = nn.Linear(self.output_dimension, self.output_dimension)
        else:
            self.mapping_model = nn.Linear(self.output_dimension, self.output_dimension)

        self.loss_fn = nn.CrossEntropyLoss()
        print("get matrix ...")
        self.matrix_v = self.get_matrix(self.matrix_v_path)
        self.matrix_t = self.get_matrix(self.matrix_t_path)

    def get_matrix(self, path):
        matrix = []
        with open(path, 'r') as f:
            count_ = 0
            reader = csv.reader(f)
            for i in reader:
                if count_ == 0:
                    count_ += 1
                    continue
                line = map(float, i)
                line = list(line)
                matrix.append(line)

        matrix = torch.tensor(matrix, requires_grad = False)
        return matrix

    def forward(self, x):
        return self.mapping_model(x)

    # def get_text_feature(self, text_vector):
    #     text_vector = text_vector.cuda()
    #     return self.forward(text_vector)

    def get_mapping_feature(self, z_support, z_text_vector):
        self.matrix_t = self.matrix_t.cuda()
        self.matrix_v = self.matrix_v.cuda()
        z_support = z_support @ self.matrix_v
        z_text_vector = z_text_vector @ self.matrix_t
        if self.seperate:
            z_s = self.mapping_v_model(z_support)
            z_t = self.mapping_t_model(z_text_vector)
        else:
            z_s = self.mapping_model(z_support)
            z_t = self.mapping_model(z_text_vector)

        return z_s, z_t

    def set_forward(self, x, text_vector):
        x = x.cuda()
        text_vector = text_vector.cuda()

        z_support = x[:, :self.n_support]
        # (n_way, vis_dimen)
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_text_vector = text_vector[:, 0, :] # (n_way, text_dimen)

        z_support, z_text_feature = self.get_mapping_feature(z_support, z_text_vector)
        # z_text_feature = self.get_text_feature(z_text_vector)

        if self.cosine:
            z_support = F.normalize(z_support, p = 2, dim = 1, eps = 1e-12)
            z_text_feature = F.normalize(z_text_feature, p = 2, dim = 1, eps = 1e-12)
            scores = cosine_dist(z_text_feature, z_support)
        else:
            dists = euclidean_dist(z_text_feature, z_support)
            scores = -dists
        return scores


    def set_forward_loss(self, x, text_vector):
        scores = self.set_forward(x, text_vector)

        gt = torch.from_numpy(np.repeat(range(self.n_way), 1))
        gt = gt.cuda()
        loss = self.loss_fn(scores, gt)
        return loss

    def train_loop(self, epoch, train_loader, optimizer, logger, logger_file):
        self.train()
        print_freq = 10

        avg_loss=0
        for i, (x, text_vector, _) in enumerate(train_loader):
            # import pdb; pdb.set_trace()
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
            # import pdb; pdb.set_trace()
            assert self.n_way  ==  x.size(0), "text_mapping do not support way change"
            correct_this, count_this = self.correct(x, text_vector)
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
    def correct(self, x, text_vector):
        scores = self.set_forward(x, text_vector)
        y_query = np.repeat(range(self.n_way), 1)

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
    # x: N * D
    # y: M * D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)


    yT = torch.transpose(y, 1, 0)

    output = x @ yT

    return output
