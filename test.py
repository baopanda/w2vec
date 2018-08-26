import gensim

path_model_w2v = 'w2v_bao_300_mincount2_new.bin'
path = "(300 dimesions)"
def LoadModel():
    print("init")
    print('Loading model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format(path_model_w2v, fvocab=None, binary=True, encoding='utf8')
    print('Ready')

    w2 = ["đồ_ăn"]
    print(path+" Most similar to {0}".format(w2), model.wv.most_similar(positive=w2, topn=6))

    print(path+" Similarity between 'đỏ' and 'đen'",
          model.wv.similarity(w1="đỏ", w2="đen"))

    print(path+" Similarity between 'đỏ' and 'xanh'",
          model.wv.similarity(w1="đỏ", w2="xanh"))

    print(path+" Similarity between 'đồ_ăn' and 'ẩm_thực'",
          model.wv.similarity(w1="đồ_ăn", w2="ẩm_thực"))

    print(path+" Similarity between 'đồ_ăn' and 'phô mai'",
          model.wv.similarity(w1="đồ_ăn", w2="phô_mai"))

    print(path+" Similarity between 'phô_mai' and 'xôi'",
          model.wv.similarity(w1="phô_mai", w2="xôi"))

    print(path+" Similarity between 'cháo' and 'xôi'",
          model.wv.similarity(w1="cháo", w2="xôi"))
    while(True):
        line = input("Nhap tu muon tim tu tuong duong: ")
        print(path+" Most similar to {0}".format(line), model.wv.most_similar(positive=line, topn=6))
        if line == 'exit':
            break


if __name__ == "__main__":
    LoadModel()