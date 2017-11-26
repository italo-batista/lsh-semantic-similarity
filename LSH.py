# coding: utf-8

from random import shuffle
from TFIDF import TFIDF
import math


class LSH(object):

    def __init__(self):
        self.dr = False

    def set_sparse_matrix(self, sparse_matrix):
        """Set the sparse matrix in which the lsh algorithm is runned."""

        self.sparse_matrix = sparse_matrix

    def set_dimensionality_reduction(self, dr):
        """Set the reduced dimensionality of sparse matrix for operations."""

        self.dr = dr

    def _get_larger_prime_number(self):

        n_cols_sparse_matrix = len(self.sparse_matrix[0])
        limit = n_cols_sparse_matrix + 50

        number = n_cols_sparse_matrix
        is_prime = self._crivo_algo(limit)
        while number < limit:
            if is_prime[number]:
                prime = number
                break
            number += 1

        return prime

    def _set_larger_prime_number(self):
        self.larger_prime = self._get_larger_prime_number()

    def _crivo_algo(self, limit):

        crivo = [True] * limit

        crivo[0] = False
        crivo[1] = False

        for i in range(int(math.sqrt(limit))):
            if crivo[i]:
                for j in range(i * i, limit, i):
                    crivo[j] = False
        return crivo

    def _get_permutation_matrix(self):

        n_cols_sparse_matrix = len(self.sparse_matrix[0])
        n_cols = n_cols_sparse_matrix
        n_rows = self.dr
        permutation_matrix = []

        self._set_larger_prime_number()

        as_coeff = [i + 1 for i in range(self.dr)]
        bs_coeff = [i + 1 for i in range(self.dr)]
        shuffle(as_coeff)
        shuffle(bs_coeff)

        for i in range(n_rows):
            a_coeff = as_coeff[i]
            b_coeff = bs_coeff[i]
            permuted_row = self._get_permuted_row(n_cols, a_coeff, b_coeff)
            permutation_matrix.append(permuted_row)

        return permutation_matrix

    def _get_permuted_row(self, n_cols, a_coeff, b_coeff):

        permuted_row = []
        for col_number in range(n_cols):
            hash_value = self.minhash(
                a_coeff, b_coeff, col_number, self.larger_prime)
            permuted_row.append(hash_value)
        return permuted_row

    def minhash(self, a_coeff, b_coeff, col_number, prime):
        hash_value = (a_coeff * col_number + b_coeff) % prime
        return hash_value

    def _get_signature_matrix(self, permutation_matrix):

        n_rows = len(self.sparse_matrix)
        n_hashes = self.dr
        sig_matrix = [[None for i in range(n_hashes)] for j in range(n_rows)]

        for i in range(n_hashes):

            hashes_values = permutation_matrix[i]
            hashes_indexs = [(index, value)
                             for index, value in enumerate(hashes_values)]
            lower_hashes = sorted(hashes_indexs, key=lambda x: x[1])

            # check if each doc has a signature for the i-th hash (1:no, 0:yes)
            has_signature_flags = [1 for t in range(n_rows)]

            for j in range(len(lower_hashes)):

                token_index = lower_hashes[j][0]
                min_hash_value = lower_hashes[j][1]

                qnt_docs = len(self.sparse_matrix)
                for d in range(qnt_docs):

                    if has_signature_flags[d] == 1:

                        doc = self.sparse_matrix[d]

                        if doc[token_index] == 1:
                            sig_matrix[d][i] = min_hash_value
                            has_signature_flags[d] = 0

                        if sum(has_signature_flags) == 0:
                            break
                if sum(has_signature_flags) == 0:
                    break

        return sig_matrix

    def _get_similarity_matrix(self, sig_matrix):

        n_docs = len(self.sparse_matrix)
        n_hashes = len(sig_matrix[0])
        similarity_matrix = [['--' for i in range(n_docs)] for j in range(n_docs)]

        for i in range(n_docs):

            doc_a = set([(sig_matrix[i][s], s) 
                for s in range(len(sig_matrix[i]))])

            for j in range(i + 1, n_docs, 1):
                
                doc_b = set([(sig_matrix[j][s], s) 
                    for s in range(len(sig_matrix[j]))])
                
                jaccard_similarity = self._get_jaccard_similarity(
                    doc_a, doc_b, n_hashes)
                
                similarity_matrix[i][j] = jaccard_similarity

        return similarity_matrix

    def _get_jaccard_similarity(self, setA, setB, n_hashes):
        sim_coef = len(setA & setB) / n_hashes
        round_coef = round(sim_coef, 2)
        return round_coef

    def get_similarity(self, matrix=None, langue=None):

        if langue == None:
            self.set_sparse_matrix(matrix)

        else:
            tfidf = TFIDF(matrix, langue)
            sparse_matrix = tfidf.get_sparse_matrix()
            self.set_sparse_matrix(sparse_matrix)

        if self.dr == False:
            dimensionality_reduction = int(round((len(matrix[0]) * 3 / 4)))
            self.set_dimensionality_reduction(dimensionality_reduction)

        permutation_matrix = self._get_permutation_matrix()
        signature_matrix = self._get_signature_matrix(permutation_matrix)
        similarity_matrix = self._get_similarity_matrix(signature_matrix)
        return similarity_matrix        
