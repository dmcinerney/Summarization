# This is a custom pytorch helper full of pytorch helper functions
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import os
import shutil
import pickle as pkl
import pdb

class ModelManipulator:
    # inputs:
    #     -model is a python Module with a forward function
    #         that takes a dictionary as inputs and returns
    #         a dictionary as outputs
    #     -optimizer must be a python optimizer
    #     -loss_function is a function must take all values
    #         in the dictionary output from model as inputs
    #         with the same name and returns a tensor
    #     -error_function has the same signature as the loss
    #         function but can return None or a tensor
    #     -grad_mod is a function that modifies the gradient
    #         on the parameters before backward is called
    #     -use_cuda is a boolean indicating whether or not
    #         to use the cuda gpu
    def __init__(self, model, optimizer, loss_function, error_function, grad_mod=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.error_function = error_function
        self.grad_mod = grad_mod
        self.device = list(model.parameters())[0].device

    def inputs_to_device(self, inputs):
        return {k:(v.to(self.device) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}

    def step(self, inputs, training=False):
        inputs = self.inputs_to_device(inputs)
        with torch.set_grad_enabled(training):
            outputs = self.model(**inputs)
            loss = self.loss_function(**outputs)
        with torch.autograd.no_grad():
            error = self.error_function(**outputs)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_mod is not None:
                self.grad_mod(self.model.parameters())
            self.optimizer.step()
        loss_value = loss.item()
        if error is not None:
            error_value = error.item()
        else:
            error_value = None
        return loss_value, error_value

    def train(self, dataset_train, batch_size, epochs, dataset_val=None, stats_every=1000, verbose_every=100, checkpoint_every=1000, checkpoint_path=None, restart=True, max_steps=None):
        tt = TrainingTracker(self, dataset_val, stats_every, verbose_every, checkpoint_every, checkpoint_path, restart, max_steps)
        i, indices_iterator = tt.initialize()
        try:
            while i < epochs:
                indices_iterator = IndicesIterator(len(dataset_train), batch_size=batch_size, shuffle=True) if indices_iterator is None else indices_iterator
                for j,indices in indices_iterator:
                    inputs = dataset_train[indices]
                    train_loss, train_error = self.step(inputs, training=True)
                    tt.step(i, j, train_loss, train_error, indices.size(0), indices_iterator)
                indices_iterator = None
                i += 1
        except StopEarlyException:
            pass

        return tt.end(i, j+1, indices_iterator)

    # batch_dim is a tuple that tells you an input key, and the batch dimension for that input
    # error function must never return None for this function to work
    def test(self, test_dataloader, batch_dim=None):
        weight = None
        loss_running_average = RunningAverage()
        error_running_average = RunningAverage()
        for inputs in test_dataloader:
            if batch_dim is not None:
                weight = inputs[batch_dim[0]].size(batch_dim[1])
            loss_value, error_value = self.step(inputs)
            loss_running_average.update(loss_value, weight)
            if error_value is not None and error_running_average is not None:
                error_running_average.update(error_value, weight)
            else:
                error_running_average = None
        return loss_running_average.average, error_running_average.average

class TrainingTracker:
    def __init__(self, model_manip, dataset_val, stats_every, verbose_every, checkpoint_every, checkpoint_path, restart, max_steps):
        self.train_steps = []
        self.train_losses = []
        self.train_errors = []
        self.validation_steps = []
        self.validation_losses = []
        self.validation_errors = []

        self.model_manip = model_manip
        self.dataset_val = dataset_val
        self.stats_every = stats_every
        self.verbose_every = verbose_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.restart = restart
        self.max_steps = max_steps

        self.step_num = 0
        self.last_step_num = -1

        if self.checkpoint_path is not None and self.restart:
            shutil.rmtree(self.checkpoint_path)
            os.mkdir(self.checkpoint_path)

    def initialize(self):
        i, indices_iterator = 0, None
        if self.checkpoint_path is not None and not self.restart:
            # get epoch
            with open(os.path.join(self.checkpoint_path, 'iternum.txt'), 'r') as iternumfile:
                i, self.step_num = eval(iternumfile.read())
            # get indices iterator
            with open(os.path.join(self.checkpoint_path, 'indices_iterator.pkl'), 'rb') as iteratorfile:
                indices_iterator = pkl.load(iteratorfile)
        return i, indices_iterator

    def step(self, i, j, train_loss, train_error, batch_size, indices_iterator):
        self.train_losses.append(train_loss)
        self.train_errors.append(train_error)
        self.train_steps.append(self.step_num)
        if self.dataset_val and self.step_num % self.stats_every == 0:
            inputs_val = random_subset(self.dataset_val, batch_size)[:]
            val_loss, val_error = self.model_manip.step(inputs_val)
            self.validation_losses.append(val_loss)
            self.validation_errors.append(val_error)
            self.validation_steps.append(self.step_num)
        if self.step_num % self.verbose_every == 0:
            # needed because error can sometimes be None
            printed_train_error = str(train_error)
            print('epoch: %i, batch: %i, train_loss: %f, train_error: %s' % (i, j, train_loss, printed_train_error))
        # check for nans in model
        param_sum = sum(p.sum() for p in self.model_manip.model.parameters())
        if param_sum != param_sum:
            raise Exception('NaNs detected in model parameters after optimizer step!')
        # save checkpoint
        if self.checkpoint_path is not None and self.step_num % self.checkpoint_every == 0:
            self.save_checkpoint(i, indices_iterator)
        self.step_num += 1
        if self.max_steps is not None and self.step_num >= self.max_steps:
            raise StopEarlyException

    def save_checkpoint(self, i, indices_iterator):
        for s in np.arange(len(self.train_steps))[np.array(self.train_steps) > self.last_step_num]:
            with open(os.path.join(self.checkpoint_path, 'train_info.txt'), 'a') as train_info:
                train_info.write(str([self.train_steps[s], self.train_losses[s], self.train_errors[s]])+'\n')
        for s in np.arange(len(self.validation_steps))[np.array(self.validation_steps) > self.last_step_num]:
            with open(os.path.join(self.checkpoint_path, 'val_info.txt'), 'a') as val_info:
                val_info.write(str([self.validation_steps[s], self.validation_losses[s], self.validation_errors[s]])+'\n')
        # save model
        torch.save(self.model_manip.model, os.path.join(self.checkpoint_path, 'model.model'))
        # save optimizer state
        with open(os.path.join(self.checkpoint_path, 'optimizer_state.pkl'), 'wb') as optimizerfile:
            pkl.dump(self.model_manip.optimizer.state_dict(), optimizerfile)
        # save epoch
        with open(os.path.join(self.checkpoint_path, 'iternum.txt'), 'w') as iternumfile:
            iternumfile.write(str([i,self.step_num]))
        # save indices iterator
        with open(os.path.join(self.checkpoint_path, 'indices_iterator.pkl'), 'wb') as iteratorfile:
            pkl.dump(indices_iterator, iteratorfile)
        self.last_step_num = self.step_num

    def end(self, i, j, indices_iterator):
        if self.checkpoint_path is not None and not (self.step_num % self.checkpoint_every == 0):
            self.save_checkpoint(i, indices_iterator)

        if self.verbose_every is not None:
            print('%i epochs with %i batches per epoch done' % (i, j))

        if self.dataset_val is not None:
            return (np.array(self.train_steps),\
                    np.array(self.train_losses),\
                    np.array(self.train_errors),),\
                   (np.array(self.validation_steps),\
                    np.array(self.validation_losses),\
                    np.array(self.validation_errors),)
        else:
            return np.array(self.train_steps),\
                   np.array(self.train_losses),\
                   np.array(self.train_errors)

class StopEarlyException(Exception):
    pass

class RunningAverage:
    def __init__(self):
        self._weight_sum = 0
        self._average = 0
    
    @property
    def average(self):
        return self._average
    
    def update(self, value, weight=1):
        self._weight_sum += weight
        self._average += weight*(value - self._average)/self._weight_sum

class PyroModelManipulator(ModelManipulator):
    def __init__(self, svi):
        self.svi = svi
        
    def step(self, inputs, training=False):
        if training:
            loss = self.svi.step(**inputs)
        else:
            loss = self.svi.evaluate_loss(**inputs)
        return loss, None

class OneAtATimeDataset(Dataset):
    def get_multiple_items(self, index_generator):
        list_of_dicts = [self[i] for i in index_generator]
        return dicts_into_batch(list_of_dicts)
    
    def get_one_item(self, index):
        raise NotImplementedError
    
    def __getitem__(self, index):
        if type(index) is slice:
            return self.get_multiple_items(range(len(self))[index])
        elif type(index) is torch.Tensor and len(index.size()) > 0:
            return self.get_multiple_items(index)
        elif type(index) is int or type(index) is torch.Tensor:
            return self.get_one_item(index)
        else:
            raise Exception
            
class VariableLength(OneAtATimeDataset):
    def get_multiple_items(self, index_generator):
        lengths = []
        data = []
        for i in index_generator:
            data.append(self.get_raw_inputs(i))
            for j,arg in enumerate(data[-1][0]):
                if j >= len(lengths):
                    lengths.append(0)
                if lengths[j] < len(arg):
                    lengths[j] = len(arg)
        return dicts_into_batch([self.prepare_inputs(v_args, nv_args, lengths) for v_args, nv_args in data])
    
    def get_one_item(self, index):
        variable_args, non_variable_args = self.get_raw_inputs(index)
        lengths = (len(arg) for arg in variable_args)
        return self.prepare_inputs(variable_args, non_variable_args, lengths)
        
    def get_raw_inputs(self, i):
        raise NotImplementedError
    
    def prepare_inputs(self, args, non_variable_args, lengths):
        raise NotImplementedError


def pack_padded_sequence_maintain_order(x, length, batch_first=False):
    length_sorted, indices = torch.sort(length, descending=True)
    invert_indices = torch.zeros_like(indices).scatter(0, indices, torch.arange(indices.shape[0], device=indices.device))
    x = pack_padded_sequence(x[indices], length_sorted, batch_first=batch_first)
    return x, invert_indices

def pad_packed_sequence_maintain_order(output, other_args, invert_indices, batch_first=False):
    output, _ = pad_packed_sequence(output, batch_first=batch_first)
    new_args = [arg[invert_indices] for arg in other_args]
    if batch_first:
        return output[invert_indices], new_args
    else:
        return output[:,invert_indices], new_args
    
def print_loop(i, length, every=1000):
    if (i+1) % every == 0 or i == length-1:
        print("%d/%d" % (i+1, length))
    
class IndicesIterator:
    def __init__(self, dataset_length, batch_size, shuffle):
        self.indices = np.arange(dataset_length)
        if shuffle:
            np.random.shuffle(self.indices)
        self.indices = torch.tensor(self.indices)
        self.batch_size = batch_size
        self.i = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if (self.i*self.batch_size) < len(self.indices):
            offset = int(self.i*self.batch_size)
            i_temp = self.i
            self.i += 1
            return i_temp, self.indices[offset:offset+self.batch_size]
        else:
            raise StopIteration
            
class MultiDatasetDataLoader:
    def __init__(self, datasets, **kwargs):
        self.dataloaders = [DataLoader(dataset, **kwargs) for dataset in datasets]
        self.indicies = np.concatenate(
            [np.zeros(len(dataloader))+i for i,dataloader in enumerate(self.dataloaders)]
        )
    
    def __iter__(self):
        self.index_iterator = self.indicies.__iter__()
        self.dataset_iterators = [dataloader.__iter__() for dataloader in self.dataloaders]
        np.random.shuffle(self.indicies)
        return self
    
    def __next__(self):
        i = int(self.index_iterator.__next__())
        return self.dataset_iterators[i].__next__()
    
    @property
    def batch_size(self):
        return self.dataloaders[0].batch_size
    
def random_subset(dataset, number_of_examples):
    subset, _ = random_split(dataset, [number_of_examples, len(dataset)-number_of_examples])
    return subset

def split_dataset(dataset, proportions):
    lengths = np.array([len(dataset)*proportion for proportion in proportions]).round().astype(np.int).tolist()
    if sum(lengths) > len(dataset):
        raise Exception
    lengths.append(len(dataset)-sum(lengths))
    return random_split(dataset, lengths)

def dicts_into_batch(list_of_dicts):
    return_dict = {}
    for dictionary in list_of_dicts:
        for key,value in dictionary.items():
            if key not in return_dict.keys():
                return_dict[key] = []
            return_dict[key].append(value.view(1,*value.size()) if isinstance(value, torch.Tensor) else value)
    for key,value in return_dict.items():
        return_dict[key] = torch.cat(value, 0) if isinstance(value[0], torch.Tensor) else value
    return return_dict

def plot_checkpoint(checkpoint_path, figure_name=None, show=True, average_over=1):
    training_values = ([], [], [])
    with open(os.path.join(checkpoint_path, 'train_info.txt'), 'r') as traininfo_file:
        for line in traininfo_file:
            step, loss, error = eval(line)
            training_values[0].append(step)
            training_values[1].append(loss)
            training_values[2].append(error)
    training_values = (np.array(v) for v in training_values)
    if os.path.exists(os.path.join(checkpoint_path, 'val_info.txt')):
        validation_values = ([], [], [])
        with open(os.path.join(checkpoint_path, 'val_info.txt'), 'r') as valinfo_file:
            for line in valinfo_file:
                step, loss, error = eval(line)
                validation_values[0].append(step)
                validation_values[1].append(loss)
                validation_values[2].append(error)
        validation_values = (np.array(v) for v in validation_values)
    else:
        validation_values = None
    figure_name = os.path.join(checkpoint_path, figure_name) if figure_name is not None else None
    return plot_learning_curves(training_values, validation_values=validation_values, figure_name=figure_name, show=show, average_over=average_over)

def plot_learning_curves(training_values, validation_values=None, figure_name=None, show=True, average_over=1):
    train_steps, train_losses, train_errors = training_values
    if validation_values is not None:
        validation_steps, validation_losses, validation_errors = validation_values
    plt.plot(smooth(train_steps, average_over), smooth(train_losses, average_over))
    if validation_values is not None:
        plt.plot(smooth(validation_steps, average_over), smooth(validation_losses, average_over))
    if figure_name is not None:
        plt.savefig(figure_name+'_loss.png')
    if show:
        plt.show()
    plt.close()
    plt.plot(smooth(train_steps, average_over), smooth(train_errors, average_over))
    if validation_values is not None:
        plt.plot(smooth(validation_steps, average_over), smooth(validation_errors, average_over))
    if figure_name is not None:
        plt.savefig(figure_name+'_error.png')
    if show:
        plt.show()
        
def smooth(array, average_over):
    if array.shape[0] == 0:
        return array
    remainder = array.shape[0] % average_over
    remainder_exists = int(remainder > 0)
    size = (array.shape[0] // average_over) + remainder_exists
    if array[0] is None:
        return array[:size]
    new_array = np.zeros(size)
    new_array[:new_array.shape[0]-remainder_exists] = array[:array.shape[0]-remainder].reshape(-1, average_over).mean(-1)
    if remainder_exists:
        new_array[new_array.shape[0]-remainder_exists:] = array[array.shape[0]-remainder:].reshape(-1, remainder).mean(-1)
    return new_array

def log_sum_exp(inputs, dim, weights=None):
    if weights is None:
        weights = torch.ones(inputs.size(), device=inputs.device)
    a = torch.max(inputs)
    return a+torch.log(torch.sum(weights*torch.exp(inputs-a), dim=dim))

def pad_and_concat(tensors, static=False):
    dim = tensors[0].dim()
    if not static:
        max_size = [0]*dim
        for tensor in tensors:
            if tensor.dim() != dim:
                raise Exception
            for i in range(dim):
                max_size[i] = max(max_size[i], tensor.size(i))
    else:
        max_size = tensors[0].size()
    concatenated_tensor = []
    for tensor in tensors:
        if not static:
            padding = []
            for i in range(dim-1,-1,-1):
                padding.extend([0,max_size[i]-tensor.size(i)])
            new_tensor = F.pad(tensor, tuple(padding))
        else:
            new_tensor = tensor
        concatenated_tensor.append(new_tensor.view(1,*new_tensor.size()))
    concatenated_tensor = torch.cat(concatenated_tensor, 0)
    return concatenated_tensor

def batch_stitch(tensor_lists, indices, static_flags=None):
    return_tensors = []
    for i,tensor_list in enumerate(tensor_lists):
        new_tensor = pad_and_concat(tensor_list, static=(False if static_flags is None else static_flags[i]))
        size = [1]*new_tensor.dim()
        size[:indices.dim()] = indices.size()
        return_tensors.append(new_tensor.gather(0, indices.view(*size).expand(size[0],*new_tensor.shape[1:])))
    return return_tensors

if __name__ == '__main__':
    print(ModelManipulator)
