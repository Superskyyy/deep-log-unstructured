"""
Do not run this on personal computers, processing will need a lot of memory and multi-core cpu power

This module import->tokenize->pickle up to 200 million rows from 10 * 20million-row data inputs
"""

from data_pickler import pickle_import

def multi_run_wrapper(args):
    return pickle_import(*args)


if __name__ == "__main__":
    thunderbird_template = '<Token0> <Token1> <Token2> <Token3> <Token4> <Token5> <Token6> <Token7> <Token8>(\[<Token9>\])?: <Message>'

    arg_list = [(thunderbird_template, f'tbird_{idx}', '-', 'tbird') for idx in range(10)]
    from multiprocessing import Pool

    pool = Pool(10)
    results = pool.map(multi_run_wrapper, arg_list)
