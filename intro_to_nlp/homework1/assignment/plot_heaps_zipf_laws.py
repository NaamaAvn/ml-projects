import json
import pickle
import os.path
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log
import seaborn as sn
import random
sn.set()


def read_data(filename):
    word2freq = defaultdict(int)

    i = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        print('reading the text file...')
        for i, line in enumerate(fin):
            for word in line.split():
                word2freq[word] += 1
            if i % 100000 == 0:
                print(i)

    total_words = sum(word2freq.values())
    word2nfreq = {w: word2freq[w]/total_words for w in word2freq}

    return word2nfreq


def plot_zipf_law(word2nfreq):
    y = sorted(word2nfreq.values(), reverse=True)
    x = list(range(1, len(y)+1))

    product = [a * b for a, b in zip(x, y)]
    # print(product[:1000])  # todo: print and note the roughly constant value

    y = [log(e, 2) for e in y]
    x = [log(e, 2) for e in x]

    plt.plot(x, y)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title("Zipf's law")
    plt.show()


def plot_heaps_law(path, step=10000, max_tokens=None):
    """
    path: path to a large text file (e.g. 'en.wikipedia2018.10M.txt')
    step: how often to record (every N tokens)
    max_tokens: optional limit to stop early
    """
    seen = set()
    N, V = [], []
    total = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                total += 1
                seen.add(word)
                if total % step == 0:
                    N.append(total)
                    V.append(len(seen))
                if max_tokens and total >= max_tokens:
                    break
            if max_tokens and total >= max_tokens:
                break
            
    plt.plot(N, V)
    plt.xlabel('Total Words N')
    plt.ylabel('Unique Words - Vocabulary Size V')
    plt.title("Heaps’ Law")
    plt.grid(True, which='both', ls=':')
    plt.show()

    plt.loglog(N, V)
    plt.xlabel('Total Words N (log)')
    plt.ylabel('Unique Words - Vocabulary Size V (log)')
    plt.title("Heaps’ Law (log-log)")
    plt.grid(True, which='both', ls=':')
    plt.show()



if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    if not os.path.isfile('word2nfreq.pkl'):
        data = read_data(config['corpus'])
        pickle.dump(data, open('word2nfreq.pkl', 'wb'))

    plot_zipf_law(pickle.load(open('word2nfreq.pkl', 'rb')))
    plot_heaps_law("data/en.wikipedia2018.10M.txt")

