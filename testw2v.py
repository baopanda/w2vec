#!/usr/bin/python
# -*- coding: utf-8 -*-
import gensim

import gensim.models as w2v
import sklearn.decomposition as dcmp
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import re
import nltk
import sys

"""
Visualize Word2Vec
Word2vec + PCA + Clustering
"""

# path_model_w2v = 'kieuc.bin'
path_model_w2v = 'data_w2v_output.bin'

class SemanticMap:
    def __init__(self, model_path):
        print("init")
        print('Loading model ...')
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, fvocab=None, binary=True, encoding='utf8')
        print('Ready')

    def __split_words(self, input_string):
        print("spilit")
        # return re.findall(r"[\w']+", input_string)
        return (input_string,)

    def __clean_words(self, words):
        print("clean")
        clean_words = []
        for w in words:
            clean_words.append(w)
        return clean_words

    def __remove_stop_words(self, words):
        print("remove")
        # return [w for w in words if not w in nltk.corpus.stopwords.words('english')]
        return words

    def __get_non_compositional_entity_vector(self, entity):
        print("non-compositional")
        print('get_non_compositional_entity_vector: ', entity)
        word = entity[0]
        print('word: ', word)
        ver_word = self.model[word]
        return ver_word

    def __get_compositional_entity_vector(self, entity):
        print("compositional")
        array = np.array(self.model[entity[0]])
        for ind in range (1, len(entity)):
            array = array + np.array(self.model[entity[ind]])
        return array/len(entity)

    def __get_vector(self, term):
        print("vector")
        words = self.__remove_stop_words(self.__clean_words(self.__split_words(term)))

        if len(words) < 1:
            print('All the terms have been filtered.')
            # raise
        if len(words) == 1:
            print(words)
            try:
                return self.__get_non_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary entity')
                raise
        elif len(words) < 4:
            try:
                return self.__get_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary word in compositional entity')
                raise
        else:
            print('Entity is too long.')

        return words
            # raise

    def __reduce_dimensionality(self, word_vectors, dimension=2):
        print("reduce")
        data = np.array(word_vectors)
        pca = dcmp.PCA(n_components=dimension)
        pca.fit(data)
        return pca.transform(data)

    def cluster_results(self, data, threshold=0.13):
        print("cluster")
        return hcluster.fclusterdata(data, threshold, criterion="distance")

    def map_words(self, words, sizes):
        print("map")
        final_words = []
        final_sizes = []
        vectors = []

        for word in words:
            try:
                vect = self.__get_vector(word)
                vectors.append(vect)
                if sizes is not None:
                    final_sizes.append(sizes[words.index(word)])
                final_words.append(word)
            except Exception:
                print('not valid ' + word)

        return vectors, final_words, final_sizes

    def plot(self, vectors, lemmas, clusters, sizes=80):
        print("plot")
        if sizes == []:
            sizes = 80
        plt.scatter(vectors[:, 0], vectors[:, 1], s=sizes, c=clusters)
        for label, x, y in zip(lemmas, vectors[:, 0], vectors[:, 1]):
            plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        plt.show()

    def map_cluster_plot(self, words, sizes, threshold):
        print("map_plot")
        vectors, words, sizes = self.map_words(words, sizes)
        vectors = self.__reduce_dimensionality(vectors)
        clusters = self.cluster_results(vectors, threshold)
        self.plot(vectors, words, clusters, sizes)

    def print_results(self, words, clusters):
        print("print")
        print(words)
        print(clusters.tolist())

    # def get_words(self,words):



def cli(mapper_cli):
    while True:
        encoding = 'utf-8' if sys.stdin.encoding in (None, 'ascii') else sys.stdin.encoding
        line = input('Enter words or MWEs > ')
        if line == 'exit':
            break
        mapper_cli.map_cluster_plot(line.split(','), None, 0.2)

if __name__ == "__main__":
    mapper = SemanticMap(path_model_w2v)
    cli(mapper)