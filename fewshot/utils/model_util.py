import os
import os.path as osp
import torch
from ..models import backbone
from ..methods import *
from .com_util import *

model_dict = dict(
    Conv4 = backbone.Conv4,
    ResNet10 = backbone.ResNet10,
    ResNet12 = backbone.ResNet12,
    ResNet18 = backbone.ResNet18,
)

def get_model(params, train=True):
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)

    if params.method == "protonet_seperately":
        few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot,
                                    text_vector_dimension = params.text_vector_dimension,
                                    output_dimension = params.output_dimension,
                                    visual_net_parameters = params.visual_net_parameters,
                                    text_net_parameters = params.text_net_parameters,
                                    result_way = params.result_way,
                                    initial = params.initial)
        if 'loss_lam' in params.keys():
            few_shot_params['loss_lam'] = params.loss_lam
        model = ProtoNet_seperately(model_dict[params.model], **few_shot_params)
    elif params.method == 'protonet_cca':
        few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot,
                            final_dimension = params.final_dimension,
                            text_vector_dimension = params.text_vector_dimension,
                            visual_net_parameters = params.visual_net_parameters,
                            matrix_v_path = params.matrix_v_path,
                            matrix_t_path = params.matrix_t_path,
                            result_way = params.result_way,
                            initial = params.initial, loss_lam = params.loss_lam)
        if 'full_layer' in params.keys():
            few_shot_params['full_layer'] = params.full_layer
        model = ProtoNet_cca(model_dict[params.model], **few_shot_params)

    model = model.cuda()
    return model

def load_model(params, modelfile):
    model = get_model(params)
    model.load_state_dict(torch.load(modelfile))
    model.cuda()
    return model.feature

def resume(params, model, optimizer):
    start_epoch = 0
    if params.resume:
        resume_file = get_resume_file(params.save_dir, params.epoch)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state_dict'])
            optimizer.load_state_dict(tmp['optimizer'])
        print('load {}'.format(params.save_dir))
        # import pdb; pdb.set_trace()
    elif params.warmup:
        baseline_save_dir = '%s/ckpts/%s/%s_%s' %('exp', params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_save_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_save_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state_dict']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    return model, optimizer, start_epoch
