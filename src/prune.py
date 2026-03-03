"""Handles all the pruning-related stuff."""
from __future__ import print_function

import collections

import numpy as np

import torch
import torch.nn as nn


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn):
        self.model = model
        self.prune_perc = prune_perc #percent of weights to prune
        self.train_bias = train_bias #wheater biases are trainable or not
        self.train_bn = train_bn #whether batchnorm params are trainable or not

        self.current_masks = None #newly computed masks after pruning of current dataset (dict storing layer masks, masks keep track which weights are related to which task/dataset)
        self.previous_masks = previous_masks #dict storing all layer masks from past, masks keep track which weights are related to which task/dataset
        valid_key = list(previous_masks.keys())[0] #determines current dataset idx
        self.current_dataset_idx = previous_masks[valid_key].max()

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda() 
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel()) #numel returns the total number of elements in the tensor
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0][0] #smallest prune_perc% will be removed

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * \
            previous_mask.eq(self.current_dataset_idx)  #.le() is '<=' || in pytorch multiplying boolean works like logical AND
                                                        # e.g case cutoff = 0.1 & weights = [0.1,0.9,0.05,0.8] & prev_mask=[data2,data2,data2,data1], 
                                                        # prev_mask.eq() line will give [T,T,T,F], after AND, remove_mask = [T,F,T,F], hence remove idx 1 and 3
        # mask = 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0 #this line will update mask where remove_mask is T(or 1) to 0, prev_mask will now be [0,data2,0,data1]
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel())) #numel() returns total no. of elements in the tensor
        return mask #return updated mask

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
        for module_idx, module in enumerate(self.model.shared.modules()): #iterating through all the layers in the shared part of the model
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear): #only prune conv and linear layers
                mask = self.pruning_mask(
                    module.weight.data, self.previous_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(
                        self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, dataset_idx): #used during inference, allows eval performance on specific dataset
        """To be done to retrieve weights just for a particular dataset."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])

    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases

    def make_finetuning_mask(self): #used when starting new dataset
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        self.current_dataset_idx += 1

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                mask[mask.eq(0)] = self.current_dataset_idx #This turns previously pruned weights into trainable weights for new dataset.
                                                            #Freed weights get reassigned to new dataset
                                                            #Old dataset weights remain frozen
        self.current_masks = self.previous_masks
