import numpy as np


def produce_glove_vector_matrix(embedding_dim, vocab_size, token):

    GLOVE_FILE = "C:\\Users\\giorg\\Documents\\Thesis\\GloveModelFile\\glove.twitter.27B."+str(embedding_dim)+"d.txt"

    glove_vectors = dict()

    file = open(GLOVE_FILE, encoding="utf-8")

    for line in file:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        glove_vectors[word] = vectors
    file.close()

    word_vector_matrix = np.zeros((vocab_size, embedding_dim))

    for word, index in token.word_index.items():
        vector = glove_vectors.get(word)
        if vector is not None:
            word_vector_matrix[index] = vector

    return word_vector_matrix
