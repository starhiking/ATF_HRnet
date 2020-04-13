import argparse
import os
import sys
import time
sys.path.append('.')

import numpy as np
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random


class MULTI_DataLoader(object):
    def __init__(self,main_loader,aux_loaders,ratios):
        """
            ratios size = aux_loader size + 1
            ratios : [5,1,2] present main_loader : aux_loader[0] : aux_loader[1] = 5:1:2
        """
        self.main_loader = main_loader
        self.aux_loaders = aux_loaders
        self.ratios = ratios / np.sum(ratios)
        self.main_iter = iter(main_loader)
        self.aux_iters = [iter(aux_loader) for aux_loader in aux_loaders]
        self.max_iter_size = np.floor(self.main_loader.sampler.num_samples / self.main_loader.batch_size).astype(np.int32) if self.main_loader.drop_last else np.ceil(self.main_loader.sampler.num_samples / self.main_loader.batch_size).astype(np.int32)
        self.current_iter = 0

    def init_iter(self):
        self.main_iter = iter(self.main_loader)
        self.current_iter = 0
        # self.aux_iters = [iter(aux_loader) for aux_loader in aux_loaders]
        
    def get_iter_flag(self):
        return self.current_iter < self.max_iter_size

    def get_iter_num(self):
        return self.current_iter
    
    def change_ratios(self,ratios):
        self.ratios = ratios / np.sum(ratios)

    def get_iter(self):
        
        if not self.get_iter_flag():
            print("Main task dataset has finished one epoch")
            return None
        
        random_seed = random.random()
        loader_output = None
        if random_seed <= self.ratios[0]:
        
            try:
                loader_output = next(self.main_iter)
                self.current_iter += 1
            except  StopIteration:
                raise("Current iter size is overflow .")

        else :
            random_seed = random_seed - self.ratios[0]
            for i in range(len(self.aux_iters)):
                if random_seed <= self.ratios[i]:
                    try:
                        loader_output = next(self.aux_iters[i])
                    except StopIteration:
                        self.aux_iters[i] = iter(self.aux_loaders[i])
                        loader_output = next(self.aux_iters[i])
                    break
                random_seed = random_seed -self.ratios[i]

        return loader_output

