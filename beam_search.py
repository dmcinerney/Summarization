class Hypothesis:
    @classmethod
    def get_top_k(cls, hypotheses, k):
        raise NotImplementedError
    
    def __init__(self):
        raise NotImplementedError
    
    def next_hypotheses(self):
        raise NotImplementedError
    
    def is_done(self):
        raise NotImplementedError

# Beam Search Model
# pass in initial hypothesese (list of hypothesis objects)
# NOTE: uses sorting at each step because a priority queue would not be worth it because
#       every time you append to a hypothesis, you get a new object
def beam_search(hypotheses, beam_size):
    if not isinstance(hypotheses[0], Hypothesis):
        raise Exception
    cls = hypotheses[0].__class__
    hypotheses = cls.get_top_k(hypotheses, beam_size)
    while True:
        # enumerates the indices of the hypotheses that are still not complete
        new_hypotheses = []
        num_not_active = 0
        for hypothesis in hypotheses:
            if not hypothesis.is_done():
                new_hypotheses += hypothesis.next_hypotheses(beam_size)
            else:
                new_hypotheses += [hypothesis]
                num_not_active += 1
        if num_not_active == len(new_hypotheses):
            results = cls.get_top_k(new_hypotheses, beam_size, sorted=True)
            break
        hypotheses = cls.get_top_k(new_hypotheses, beam_size)
    return results