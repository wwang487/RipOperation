def compute_acc(rec, prc):
    '''
    rec： recall
    prc： precision
    '''
    acc = 2 * rec * prc / (rec + prc)
    return acc
    