# This is a custom pytorch helper full of pytorch helper functions
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

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
    def __init__(self, model, optimizer, loss_function, error_function, grad_mod=None, use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.error_function = error_function
        self.grad_mod = grad_mod
        self.use_cuda = use_cuda
        if self.use_cuda:
            if not torch.cuda.is_available():
                raise Exception
            self.model.cuda()
    
    def inputs_to_cuda(self, inputs):
        if self.use_cuda:
            return {k:(v.cuda() if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
        else:
            return inputs
        
    def step(self, inputs, training=False):
        inputs = self.inputs_to_cuda(inputs)
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

    def train(self, train_dataloader, epochs, dataset_val=None, stats_every=1000, verbose_every=100):
        train_steps, train_losses, train_errors, validation_steps, validation_losses, validation_errors = [], [], [], [], [], []
        step = 0
        for i in range(epochs):
            for j,inputs in enumerate(train_dataloader):
                train_loss, train_error = self.step(inputs, training=True)
                train_losses.append(train_loss)
                train_errors.append(train_error)
                train_steps.append(step)
                if dataset_val and step % stats_every == 0:
                    inputs_val = random_subset(dataset_val, train_dataloader.batch_size)[:]
                    val_loss, val_error = self.step(inputs_val)
                    validation_losses.append(val_loss)
                    validation_errors.append(val_error)
                    validation_steps.append(step)
                if step % verbose_every == 0:
                    # needed because error can sometimes be None
                    printed_train_error = str(train_error)
                    print("epoch: %i, batch: %i, train_loss: %f, train_error: %s" % (i, j, train_loss, printed_train_error))
                step += 1

        if verbose_every:
            print("%i epochs with %i batches per epoch done" % (i+1, j+1))

        if dataset_val:
            return (np.array(train_steps),\
                    np.array(train_losses),\
                    np.array(train_errors),),\
                   (np.array(validation_steps),\
                    np.array(validation_losses),\
                    np.array(validation_errors),)
        else:
            return np.array(train_steps),\
                   np.array(train_losses),\
                   np.array(train_errors)
    
    
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
    invert_indices = torch.arange(indices.shape[0])[indices]
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

# loads batches that are of different lengths
class VariableBatchDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        indices = torch.tensor(self.indices)
        for i in range(len(indices)//self.batch_size):
            offset = int(i*self.batch_size)
            yield self.dataset[indices[offset:offset+self.batch_size]]
            
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

def plot_learning_curves(training_values, validation_values=None, figure_name=None, show=True):
    train_steps, train_losses, train_errors = training_values
    if validation_values is not None:
        validation_steps, validation_losses, validation_errors = validation_values
    plt.plot(train_steps, train_losses)
    if validation_values is not None:
        plt.plot(validation_steps, validation_losses)
    if figure_name is not None:
        plt.savefig(figure_name+'_loss.png')
    plt.show()
    plt.plot(train_steps, train_errors)
    if validation_values is not None:
        plt.plot(validation_steps, validation_errors)
    if figure_name is not None:
        plt.savefig(figure_name+'_error.png')
    if show:
        plt.show()
    
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