import json
import time


def kmeans_cluster_and_evaluate(data_file, encoding_type, invocations):
    # todo: implement this function
    print(f'starting kmeans clustering and evaluation with {data_file} and {encoding_type}')

    # todo: perform feature extraction from sentences and
    #  write your own kmeans implementation with random (or KMeans++) centroid initialization

    # todo: evaluate against known ground-truth with RI and ARI:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html and
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    # todo: fill in the dictionary below with evaluation scores averaged over X invocations
    evaluation_results = {'mean_RI_score':  0.0,
                          'mean_ARI_score': 0.0}

    return evaluation_results


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'],
                                          config["encoding_type"],
                                          config["invocations"])

    for k, v in results.items():
        print(k, v)

    print(f'total time: {round(time.time()-start, 0)} sec')
