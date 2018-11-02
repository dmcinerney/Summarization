import torch
from pytorch_helper import pad_and_concat, batch_stitch
from beam_search import Hypothesis
import pdb

# Description: This file contains helper classes and functions for the summarization models
# Outline:
# a) GeneratedSummary (used in generatring summaries during test time)
# b) GeneratedSummaryHypothesis (used for beam search and subclasses Hypothesis and wraps GeneratedSummary object)
# c) PointerInfo (used in pointer-generator model to keep track of extra info used by this model) (kind of hacky)
# d) loss_function (this is trivial but is used by ModelManipulator in pytorch_helper.py)
# e) error_function (this is also trivial but is used by ModelManipulator in pytorch_helper.py)

class GeneratedSummary:
    @classmethod
    def batch_stitch(cls, generated_summaries, indices):
        # concatenate all of the relevant attributes into a list (all elements should be tensors)
        summary_list = [gs.summary for gs in generated_summaries]
        summary_length_list = [gs.summary_length for gs in generated_summaries]
        loss_unnormalized_list = [gs.loss_unnormalized for gs in generated_summaries]
        extras_list = []
        for i in range(len(generated_summaries[0].extras)):
            extras_list.append([gs.extras[i] for gs in generated_summaries])
        
        # use batch_stitch to get the resultant attribute tensors of size (indices.size(0), batch_length, etc...)
        summary_list, summary_length_list, loss_unnormalized_list = batch_stitch(
            [summary_list,
             summary_length_list,
             loss_unnormalized_list],
            indices,
            static_flags=[False, True, True]
        )
        extras_list = batch_stitch(
            extras_list,
            indices,
            static_flags=[False]*len(extras_list)
        )
        
        curr_length = max(len(gs) for gs in generated_summaries)
        new_generated_summaries = []
        for i in range(indices.size(0)):
            # reduce padding to only what is needed (if summary is not done, then this is just the current length)
            max_length = torch.max(summary_length_list[i]) if not (summary_length_list[i] == -1).any() else curr_length
            # create new summary and set the state to the state of the stitched batch
            new_generated_summary = GeneratedSummary(
                end_index=generated_summaries[0].end_index,
                summary=summary_list[i, :, :max_length],
                summary_length=summary_length_list[i],
                loss_unnormalized=loss_unnormalized_list[i],
                extras=[extra_list[i, :, :(max_length-1)] for extra_list in extras_list]
            )
            
            new_generated_summaries.append(new_generated_summary)
            
        return new_generated_summaries
    
    def __init__(self, batch_length=None, device=None, start_index=None, end_index=None, summary=None, summary_length=None, valid_indices=None, loss_unnormalized=None, extras=None):
        self.summary = torch.zeros((batch_length,1), device=device).long()+start_index if summary is None else summary
        self.summary_length = (torch.zeros(batch_length, device=device).long()-1) if summary_length is None else summary_length
        self.valid_indices = torch.arange(self.summary_length.size(0), device=self.summary_length.device)[self.summary_length < 0]
        self.end_index = end_index
        self.loss_unnormalized = torch.zeros(batch_length, device=device) if loss_unnormalized is None else loss_unnormalized
        self.extras = [] if extras is None else extras
        
    def get_summary_t(self):
        return self.summary[:,-1], self.valid_indices
    
    def update(self, summary_tp1, loss_t, extras):
        self.summary = torch.cat((self.summary, summary_tp1.unsqueeze(-1)), -1)
        
        # add loss for each batch
        self.loss_unnormalized += loss_t
        
        # get indices of instances that are not finished
        # and get indices of instances that are finished
        ending = (summary_tp1[self.valid_indices] == self.end_index)
        ended_indices = self.valid_indices[ending]
        self.valid_indices = self.valid_indices[ending == 0]
        
        # set summary length for ended time steps
        self.summary_length[ended_indices] = self.summary.size(1)
        
        for i,extra in enumerate(extras):
            if len(self) == 2:
                self.extras.append(torch.zeros(0, device=extra.device))
            self.extras[i] = torch.cat((self.extras[i], extra), 1)
        
    def loss(self):
        return self.loss_unnormalized/(self.length().float()-1)
        
    def is_done(self):
        return (self.summary_length >= 0).sum() == self.summary_length.size(0) or len(self) > 300
        
    def return_info(self):
        extras = (extra.cpu().detach().numpy() for extra in self.extras)
        return (self.summary.cpu().detach().numpy(), self.summary_length.cpu().detach().numpy(), *extras)
    
    def length(self):
        length = torch.tensor(self.summary_length)
        length[length < 0] = len(self)
        return length
    
    def __len__(self):
        return self.summary.size(1)
    
    def copy(self):
        gs_copy = GeneratedSummary(
            end_index=self.end_index,
            summary=torch.tensor(self.summary),
            summary_length=torch.tensor(self.summary_length),
            loss_unnormalized=torch.tensor(self.loss_unnormalized),
            extras=[torch.tensor(extra) for extra in self.extras]
        )
        return gs_copy
    
class GeneratedSummaryHypothesis(Hypothesis):
    @classmethod
    def get_top_k(cls, hypotheses, k, sorted=False):
        losses = pad_and_concat([hyp.generated_summary.loss() for hyp in hypotheses], static=True)
        indices = torch.topk(losses, k, dim=0, largest=False, sorted=sorted)[1]
        return cls.batch_stitch(hypotheses, indices)
    
    @classmethod
    def batch_stitch(cls, hypotheses, indices):
        # set attributes that stay the same
        model, text_states, text_length = hypotheses[0].model, hypotheses[0].text_states, hypotheses[0].text_length
        
        # call the stitch function on all non-tensor attributes that differ
        generated_summaries = GeneratedSummary.batch_stitch([hyp.generated_summary for hyp in hypotheses], indices)
        
        # create tensors of all of the tensor attributes that differ
        h_list = [hyp.h for hyp in hypotheses]
        c_list = [hyp.c for hyp in hypotheses]
        coverage_list = [hyp.coverage for hyp in hypotheses]
        h_list, c_list, coverage_list = batch_stitch(
            [h_list, c_list, coverage_list],
            indices,
            static_flags=[True, True, True]
        )
        
        return [cls(model, generated_summaries[i], text_states, text_length, h_list[i], c_list[i], coverage_list[i]) for i in range(h_list.size(0))]
        
    def __init__(self, model, generated_summary, text_states, text_length, h, c, coverage):
        self.model = model
        self.generated_summary = generated_summary
        self.text_states = text_states
        self.text_length = text_length
        self.h, self.c = h, c
        self.coverage = coverage
        
        self.batch_length = text_states.size(0)
        self.device = text_states.device
        
    def next_hypotheses(self, beam_size):
        # set timestep words, valid indices
        summary_t, valid_indices = self.generated_summary.get_summary_t()

        # take a time step
        vocab_dist, self.h, self.c, attention, _ = self.model.timestep(valid_indices, summary_t, self.text_states, self.text_length, self.h, self.c, self.coverage)
        
        hypotheses = []
        word_indices = torch.topk(vocab_dist, beam_size, dim=1)[1]
        for i in range(beam_size):
            generated_summary_temp = self.generated_summary.copy()
            
            # generate next summary words
            summary_tp1 = torch.zeros(self.batch_length, device=self.device).long()
            summary_tp1[valid_indices] = word_indices[:,i]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = torch.zeros(self.batch_length, device=self.device)
            log_prob[valid_indices] = self.model.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.model.with_coverage:
                covloss = torch.zeros(self.batch_length, device=self.device)
                covloss[valid_indices] = self.model.calculate_covloss(self.coverage[valid_indices], attention[valid_indices])
                self.coverage += attention
            
            # update global log prob
            loss_t = -log_prob + self.model.gamma*(covloss if self.model.with_coverage else 0)
            
            # trick so that duplicate batch examples aren't chosen in the top k
            if i > 0:
                loss_t[generated_summary_temp.summary_length > 0] = float('inf')
            
            # get any extra things the model wants to store in your summary object
            extras = (log_prob.unsqueeze(-1), attention, *self.model.get_extras())
            
            # update summary
            generated_summary_temp.update(summary_tp1, loss_t, extras)
            
            # add this generated summary as another hypothesis
            hyp = GeneratedSummaryHypothesis(self.model, generated_summary_temp, self.text_states, self.text_length, self.h, self.c, self.coverage)
            hypotheses.append(hyp)
            
        return hypotheses
    
    def is_done(self):
        return self.generated_summary.is_done()
    
        
class PointerInfo:
    def __init__(self, text, text_oov_indices):
        self.text = text
        self.text_oov_indices = text_oov_indices
        self.oov_lengths = torch.tensor([len(oovs) for oovs in text_oov_indices], device=text.device).int()
        self.max_num_oov = torch.max(self.oov_lengths)
        self.word_indices = torch.unique(text)
        self.valid_indices = None
        self.current_p_gen = None
        
    def update_valid_indices(self, valid_indices):
        self.valid_indices = valid_indices
        
    def update_p_gen(self, p_gen):
        self.current_p_gen = torch.zeros((self.text.size(0),1), device=self.text.device).scatter(0, self.valid_indices.unsqueeze(-1), p_gen)
        
    def get_text(self):
        return self.text[self.valid_indices] if self.valid_indices is not None else text
    
    def get_oov_lengths(self):
        return self.oov_lengths[self.valid_indices] if self.valid_indices is not None else self.oov_lengths
    
def loss_function(loss):
    return loss.sum()

def error_function(loss):
    return None