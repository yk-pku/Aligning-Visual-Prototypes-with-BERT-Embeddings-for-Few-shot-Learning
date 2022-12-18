import os
import os.path as osp
import torch
import glob
import numpy as np

def get_save_dir(params):
    save_dir = '%s/ckpts/%s/%s_%s' %(params.save_dir, params.dataset, params.model, params.method)
    if 'text_vector_type' in params.keys():
        # if params.text_vector_type != 'GloVe': # GloVe is the default option for jointly training
        #     save_dir += params.text_vector_type
        save_dir += params.text_vector_type
    if 'experiment_number' in params.keys():
        save_dir += params.experiment_number
    if params.train_aug:
        save_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']:
        save_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_resume_file(save_dir, epoch):
    # filelist = glob.glob(os.path.join(save_dir, '*.tar'))
    # if len(filelist) == 0:
    #     return None
    #
    # filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    # epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    # max_epoch = np.max(epochs)
    # resume_file = os.path.join(save_dir, '{:d}.tar'.format(max_epoch))

    # resume_file = os.path.join(save_dir, 'best_model.tar')
    resume_file = os.path.join(save_dir, '{:d}.tar'.format(epoch))

    return resume_file

def get_assigned_file(save_dir, num):
    assign_file = os.path.join(save_dir, '{:d}.tar'.format(num))
    return assign_file

def get_best_file(save_dir):
    best_file = os.path.join(save_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(save_dir)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))

    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  )

    return np.mean(cl_sparsity)
