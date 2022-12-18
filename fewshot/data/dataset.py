# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, text_vector_path = None):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        if text_vector_path is not None:
            self.label_to_vector = {}
            with open(text_vector_path, 'r') as f:
                for line in f:
                    line = line.rstrip('\r\n')
                    data_line = json.loads(line)
                    # self.label_to_vector[data_line.keys()[0]]
                    for key, value in data_line.items():
                        self.label_to_vector[key] =  value

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.cl_to_label = {}
        # for cl, value in zip(self.cl_list, self.meta['label_names']):
        #     self.cl_to_label[cl] = value
        for cl in self.cl_list:
            self.cl_to_label[cl] = self.meta['label_names'][cl]

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        # import pdb; pdb.set_trace()

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)
        for cl in self.cl_list:
            if text_vector_path is not None:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform, cl_to_label = self.cl_to_label, label_to_vector = self.label_to_vector)
            else:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )


    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, cl_to_label = None, label_to_vector = None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.text = False if cl_to_label is None else True
        if self.text:
            self.cl_to_label = cl_to_label
            self.label_to_vector = label_to_vector
            for key, value in self.label_to_vector.items():
                self.text_dimension = len(value)
                break

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)

        judge_label = self.sub_meta[i].split('/')[-2]

        # add text information
        if self.text:
            label = self.cl_to_label[self.cl]
            assert label == judge_label, 'label is {}, while judge_label is {}'.format(label, judge_label)
            if label not in self.label_to_vector:
                text_vector = torch.randn(self.text_dimension)
            else:
                text_vector = self.label_to_vector[label]
                text_vector = torch.Tensor(text_vector)
            return img, text_vector, target

        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
