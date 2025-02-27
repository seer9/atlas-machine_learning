#!/usr/bin/env python3
"""early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    cost: current validation cost
    opt_cost: lowest recorded validation cost
    threshold: threshold used for early stopping
    patience: the patience count used for early stopping
    count: the count of how long the threshold has not been met
    """
    if (opt_cost - cost) <= threshold:
        count += 1
    else:
        count = 0

    return (count >= patience, count)
