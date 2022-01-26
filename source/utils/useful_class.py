from itertools import product
from typing import Dict, List


class ParameterGrid:
    """ Grid of parameters with a discrete number of values for each.
    Original: https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/model_selection/_search.py#L50
    :parameter:
    param_grid = {
            'delta': [0.1, 0.2, 0.3],
             'gamma': [1, 10]
             }
    """

    def __init__(self, param_grid: Dict[str, List[float]]):
        self.param_grid = [param_grid]

    def __iter__(self):
        """Iterate over the points in the grid.
              Returns
              -------
              params : iterator over dict of str to any
                  Yields dictionaries mapping each estimator parameter to one of its
                  allowed values.
              """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
