import torch
import argparse
import yaml
from easydict import EasyDict
import sys
import os
import glob
import numpy as np
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
from fewshot.utils import com_util, data_util, model_util
from tensorboardX import SummaryWriter

# if need, set GPU device number
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(params):
    np.random.seed(10)

    print('dataset: {}, method: {}, model: {}'.format(params.dataset, params.method, params.model))

    # dataset
    base_loader, val_loader = data_util.get(params)

    # model
    model = model_util.get_model(params)

    # optimizer
    optim = params.optim.lower()
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.base_lr,
                         momentum=params.momentum, weight_decay=params.weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.base_lr,
                         weight_decay=params.weight_decay)

    # resume
    params_temp = params.copy()
    params_temp = EasyDict(params_temp)
    params_temp.save_dir = com_util.get_save_dir(params_temp)
    model, optimizer, start_epoch = model_util.resume(params_temp, model, optimizer)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.steps,
                         gamma=params.gamma, last_epoch=start_epoch-1)

    stop_epoch = params.stop_epoch
    train(base_loader, val_loader, model, lr_scheduler, start_epoch, stop_epoch, params)


def train(base_loader, val_loader, model, lr_scheduler, start_epoch, stop_epoch, params):
    best_acc = 0
    params.save_dir = com_util.get_save_dir(params)

    logger = SummaryWriter(params.save_dir)
    logger_file_path = os.path.join(params.save_dir, 'logger_file.txt')
    logger_file = open(logger_file_path, 'w')

    # epoch loop
    for epoch in range(start_epoch, stop_epoch):
        lr_scheduler.step()
        # train
        model.train_loop(epoch, base_loader, lr_scheduler.optimizer, logger, logger_file)
        # val
        acc = model.test_loop(val_loader, logger_file)

        logger.add_scalar('acc', acc, epoch + 1)
        # save best
        if acc > best_acc:
            best_acc = acc
            outfile = os.path.join(params.save_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer': lr_scheduler.optimizer.state_dict(),
                     'lr': lr_scheduler.get_lr()[0]}, outfile)
        # save model
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1) :
            outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer': lr_scheduler.optimizer.state_dict(),
                     'lr': lr_scheduler.get_lr()[0]}, outfile)

    logger_file.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='cfgs/baseline/miniImagenet.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config['common'])
    params.update(config['train'])

    main(params)
