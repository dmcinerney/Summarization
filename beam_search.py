class Hypothesis:
    @staticmethod
    def get_top_k(hypotheses, k):
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
    if isinstance(hypotheses[0], Hypothesis):
        raise Exception
    hypotheses = Hypothesis.get_top_k(hypotheses, beam_size)
    while True:
        # enumerates the indices of the hypotheses that are still not complete
        new_hypotheses = []
        num_not_active = 0
        for hyp in hypotheses:
            if not hyp.is_done():
                new_hypotheses += hypothesis.next_hypotheses()
            else:
                new_hypotheses += [hypothesis]
                num_not_active += 1
        if num_not_active == len(new_hypotheses):
            result = Hypothesis.get_top_k(new_hypotheses, 1)[0]
            break
        hypotheses = Hypothesis.get_top_k(new_hypotheses, beam_size)
    return result