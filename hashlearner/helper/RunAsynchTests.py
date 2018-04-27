from multiprocessing.pool import Pool
from itertools import repeat


def run_asynch_test(function, dataset, iterations, poolsize=10):
    p = Pool(poolsize)
    iteration_list = list(range(0, iterations))
    dataset_list = list(repeat(dataset, iterations))
    results = p.starmap(function, zip(dataset_list, iteration_list))
    trained = [result[0] for result in results]
    random = [result[1] for result in results]
    return trained,random
