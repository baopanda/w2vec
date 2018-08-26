import gzip
import gensim
import logging
import os

# pathModelTxt='c.txt'

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

#DataSet tren reviews Hotels
def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


#Read thanhf 1 list de chuyen sang Word2Vec model
def read_input(input_file):
    logging.info("reading file {0}".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10000 == 0):
                logging.info("read {0} lines".format(i))
            #  pre-processing (tokens va lowercasing etc) sau do tra ve list of words
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "data_bao_final_output_truong.txt")

    # read  tokenized va chuyen ve listst
    # moi review se thanh cac serries words
    # vi the se thanh list of list
    documents = list(read_input(data_file))
    logging.info("Done reading file")

    # bd train
    model = gensim.models.Word2Vec(
        documents,
        size=300, #size vector
        window=10, #max diatance btw target va neighbours trai va phai
        min_count=2, #min frequency
        workers=5, #no.of threads
        iter = 5,
        cbow_mean=1,
        negative=5,
        sg=0,
        sample=0.0001
    )
    model.train(documents, total_examples=len(documents), epochs=10)

    # save word vector
    # model.wv.save(os.path.join(abspath, "data_w2v_output1.txt"))
    model.wv.save_word2vec_format("w2v_bao_300_mincount2_new.txt", fvocab=None, binary=False)
    model.wv.save_word2vec_format("w2v_bao_300_mincount2_new.bin", fvocab=None, binary=True)
    # model.wv.save()
    # w1 = "dirty"
    # print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))
    #
    #
    # w1 = ["pizza"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))

    # # w1 = ["france"]
    # # print(
    # #     "Most similar to {0}".format(w1),
    # #     model.wv.most_similar(
    # #         positive=w1,
    # #         topn=6))
    #
    #
    # w1 = ["shocked"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    #
    # w1 = ["beautiful"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    #
    # w1 = ["bed", 'sheet', 'pillow']
    # w2 = ['couch']
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         negative=w2,
    #         topn=10))
    #
    #
    # print(model.score())
    print("(100 dimesions)Similarity between 'đỏ' and 'đen'",
          model.wv.similarity(w1="đỏ", w2="đen"))

    print("(100 dimesions)Similarity between 'đỏ' and 'xanh'",
          model.wv.similarity(w1="đỏ", w2="xanh"))

    print("(100 dimesions)Similarity between 'đồ_ăn' and 'ẩm_thực'",
          model.wv.similarity(w1="đồ_ăn", w2="ẩm_thực"))

    print("(100 dimesions)Similarity between 'đồ_ăn' and 'phô mai'",
          model.wv.similarity(w1="đồ_ăn", w2="phô_mai"))

    print("(100 dimesions)Similarity between 'phô_mai' and 'xôi'",
          model.wv.similarity(w1="phô_mai", w2="xôi"))

    print("(100 dimesions)Similarity between 'cháo' and 'xôi'",
          model.wv.similarity(w1="cháo", w2="xôi"))
