from model_helpers import *

class HierarchicalGeneratedSummaryHypothesis(Hypothesis):
    @classmethod
    def get_top_k(cls, hypotheses, k, sorted=False):
        losses = pad_and_concat([hyp.generated_summary.loss() for hyp in hypotheses], static=True)
        indices = torch.topk(losses, k, dim=0, largest=False, sorted=sorted)[1]
        return cls.batch_stitch(hypotheses, indices)

    @classmethod
    def batch_stitch(cls, hypotheses, indices):
        # set attributes that stay the same
        model, text_states, sentence_states, text_length = hypotheses[0].model, hypotheses[0].text_states, hypotheses[0].sentence_states, hypotheses[0].text_length

        # call the stitch function on all non-tensor attributes that differ
        generated_summaries = GeneratedSummary.batch_stitch([hyp.generated_summary for hyp in hypotheses], indices)

        # create tensors of all of the tensor attributes that differ
        coverage_list = [hyp.coverage for hyp in hypotheses]
        context_vector_list = [hyp.context_vector for hyp in hypotheses]
        (coverage_list, context_vector_list) = batch_stitch(
            [coverage_list, context_vector_list],
            indices,
            static_flags=[True, True]
        )
        if hypotheses[0].state is None:
            # we can do this bc for this object, if one is None then all will be None
            state_list = [None for _ in range(indices.size(0))]
        else:
            state_list = [hyp.state for hyp in hypotheses]
            (state_list,) = batch_stitch(
                [state_list],
                indices,
                static_flags=[False]
            )

        return [cls(model, generated_summaries[i], text_states, sentence_states, text_length, state_list[i], coverage_list[i], context_vector_list[i]) for i in range(indices.size(0))]

    def __init__(self, model, generated_summary, text_states, sentence_states, text_length, state, coverage, context_vector):
        self.model = model
        self.generated_summary = generated_summary
        self.text_states = text_states
        self.sentence_states = sentence_states
        self.text_length = text_length
        self.state = state
        self.coverage = coverage
        self.context_vector = context_vector

        self.batch_length = text_states.size(0)
        self.device = text_states.device

    def next_hypotheses(self, beam_size):
        # set timestep words, valid indices
        summary_t, valid_indices = self.generated_summary.get_summary_t()

        # take a time step
        vocab_dist, self.state, attention, self.context_vector = self.model.timestep_wrapper(valid_indices, summary_t, self.text_states, self.sentence_states, self.text_length, self.state, self.coverage, self.context_vector)

        hypotheses = []
        word_indices = torch.topk(vocab_dist, beam_size, dim=1)[1]
        for i in range(beam_size):
            generated_summary_temp = self.generated_summary.copy()

            # generate next summary words
            summary_tp1 = torch.zeros(self.batch_length, device=self.device).int()
            summary_tp1[valid_indices] = word_indices[:,i].int()

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
            # NOTE: there are duplicates for examples in an unfinished batch that
            #       have already finished and are not being altered
            if i > 0:
                loss_t[generated_summary_temp.summary_length > 0] = float('inf')

            # get any extra things the model wants to store in your summary object
            extras = (log_prob.unsqueeze(1), attention.unsqueeze(1), *self.model.get_extras())

            # update summary
            generated_summary_temp.update(summary_tp1, loss_t, extras)

            # add this generated summary as another hypothesis
            hyp = HierarchicalGeneratedSummaryHypothesis(self.model, generated_summary_temp, self.text_states, self.sentence_states, self.text_length, self.state, self.coverage, self.context_vector)
            hypotheses.append(hyp)

        return hypotheses

    def is_done(self):
        return self.generated_summary.is_done()
