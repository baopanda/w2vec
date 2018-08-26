import gzip
import gensim
import logging
import os

# pathModelTxt='c.txt'

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

#DataSet is on reviews of Hotels
def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


#Read it into a list so that we can pass this to Word2Vec model
def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing (tokens and lowercasing etc) and return list of words(tokens) for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "truyen_kieu_data_result.txt")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150, #size of vector representing each word.
        window=10, #max diatance btw target and neighbours left and right
        min_count=2, #min frequency
        workers=10) #no.of threads
    model.train(documents, total_examples=len(documents), epochs=10)

    # save only the word vectors
    model.wv.save(os.path.join(abspath, "kieua.txt"))
    model.wv.save_word2vec_format("kieub.txt", fvocab=None, binary=False)
    model.wv.save_word2vec_format("kieuc.bin", fvocab=None, binary=True)

    w1 = "thắm"
    print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1)) #returns top 10

    # look up top 6 words similar to 'polite'
    w1 = ["người"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))
    #
    # # look up top 6 words similar to 'france'
    # # w1 = ["france"]
    # # print(
    # #     "Most similar to {0}".format(w1),
    # #     model.wv.most_similar(
    # #         positive=w1,
    # #         topn=6))
    #
    # # look up top 6 words similar to 'shocked'
    # w1 = ["shocked"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    # # look up top 6 words similar to 'beautiful'
    # w1 = ["beautiful"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    # # get everything related to stuff on the bed
    # w1 = ["bed", 'sheet', 'pillow']
    # w2 = ['couch']
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         negative=w2,
    #         topn=10))
    #
    # # similarity between two different words
    # print("Similarity between 'dirty' and 'smelly'",
    #       model.wv.similarity(w1="dirty", w2="smelly"))
    #
    # similarity between two identical words
    print("Similarity between 'mưa' and 'nắng'",
          model.wv.similarity(w1="mưa", w2="nắng"))

    # similarity between two unrelated words
    print("Similarity between 'mưa' and 'mưa'",
          model.wv.similarity(w1="mưa", w2="mưa"))
