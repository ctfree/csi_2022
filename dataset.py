import os, sys, logging
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from glob import glob
from functools import partial
from itertools import islice

import util
import mautil as mu

import torch
import random


logger = logging.getLogger(__name__)




 

class DatasetMix():
    def __init__(self, cfg, data_type, data):
        self.cfg = cfg
        self.phase=data_type
        self.data_type = data_type
        self.data = data
        self.inds = np.arange(len(data))

    def __iter__(self):
        if self.data_type=='train':
            np.random.shuffle(self.inds)
        if self.data_type =='train':
            inds = np.concatenate([self.inds]*self.cfg.n_repeat)
        else:
            inds = self.inds
        for batch_id in np.arange(0, len(inds), self.cfg.batch_size):
            batch_inds = inds[batch_id:batch_id+self.cfg.batch_size]
            if len(batch_inds)!=self.cfg.batch_size and self.data_type=='train':
                break
            else:
                yield self.gen_batch(batch_inds)

    def __getitem__(self, index):
        item = self.getitem(index)
        item = self.item2tensor(item)
        return item

    def mixup(self, index):
        y = self.data[index]

        # if self.phase == 'train' and random.random() < 0.3:
        #     index_ = random.randint(0, self.data.shape[0] // 100 - 1) * 100 + index % 100
        #     p = random.random()
        #     # rows = max(int(128 * p), 20)
        #     _rows = [i for i in range(128)]
        #     random.shuffle(_rows)
        #     # _rows = _rows[:128]
        #     y[_rows] =
        if self.phase == 'train' and random.random() < 1:
            p = random.random()
            dev = (y-0.5)**2*0.1*p
            # if p<0.8:
            #     y=y-dev
            # else:
            # y=y-0.02*p*y+dev
            y=y+dev
            # return y
        # if self.phase == 'train' and random.random() < 0.5:
        #     y = y[:, ::-1, :].copy()
        # if self.phase == 'train' and random.random() < 0.5:
        #     y = 1 - self.data[index]  # 数据中存在类似正交的关系
        # if self.phase == 'train' and random.random() < 0.5:
        #     _ = y
        #     _[:, :, 0] = y[:, :, 1]
        #     _[:, :, 1] = y[:, :, 0]
        #     y = _  # 不同时刻数据实虚存在部分相等的情况
        if self.phase == 'train' and random.random() < 0.8:
            index_ = random.randint(0, self.data.shape[0] // 100 - 1) * 100 + index % 100
            p = random.random()
            rows = max(int(128 * p), 20)
            _rows = [i for i in range(128)]
            random.shuffle(_rows)
            _rows = _rows[:rows]
            # print(_rows)
            if random.random() < 1:
                y[_rows] = self.data[index_][_rows]  # 不同采样点按行合并，保持采样点独有特性，减轻模型对24那个维度的依赖
            else:
                y = (1 - p * 0.2) * y + (p * 0.2) * self.data[index_]  # 增加数值扰动，保持采样点独有特性
        return y
        
    def getitem(self, index):
        # item = {'csi': self.data[index]}
        if self.data_type=='train':
            item = {'csi': self.mixup(index)}
        else:
            item = {'csi': self.data[index]}
        return item

    def gen_batch(self, inds):
        data = self.data[inds]
        batch = {'csi': torch.tensor(data)}
        return batch

    def item2tensor(self, item):
        for k, v in item.items():
            item[k] = torch.tensor(v)
        return item


class Dataset(DatasetMix, torch.utils.data.Dataset):
    def __len__(self):
        return len(self.data)


def collate_fn(batch, mixup=0, data_type='train'):
    batch = {k: torch.stack([dic[k] for dic in batch]) for k in batch[0]}
    num = int(batch['csi'].shape[0]*mixup)
    if num>1:
        x1 = batch['csi'][:num]
        x2 = torch.cat([x1[1:], x1[:1]], 0)
        w = torch.tensor(np.random.beta(1, 1, (x1.shape[0], 1, 1, 1)).astype(np.float32))
        batch['csi'] = torch.cat([w*x1 + (1-w)*x2, batch['csi'][num:]], 0)
    return batch


def gen_ds(args, data_type, data):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        drop_last, shuffle = True, True
        collate_func = partial(collate_fn, mixup=args.mixup, data_type=data_type)
    ds = Dataset(args, data_type, data)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_func)
    return dl


if __name__ == '__main__':
    import mautil as mu
    from util import *
    args = parser.parse_args([])
    args.dataname = 'csi'
    args.batch_size=8
    args.is_debug=True
    args.n_dl_worker=1
    data = util.load_data(args)
    dl = gen_ds(args, 'train', data)
    for batch in tqdm(dl):
        pass



