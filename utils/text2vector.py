from gensim.models import word2vec
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def train_dict(texts_cut=None, sg=1, size=128, window=5, min_count=1):
    """
    训练词向量模型词典

    :param texts_cut: list of list of str, 文本的分词结果
    :param sg: int, 1表示Skip-gram，默认为1
    :param size: int, 特征向量的维度，默认为128
    :param window: int, 上下文窗口大小，表示当前词和预测词可能的最大距离，默认为5
    :param min_count: int, 忽略频率低于此值的词，默认为1
    :return: gensim.models.word2vec.Word2Vec对象，训练得到的Word2Vec模型
    """
    model_word2vec = word2vec.Word2Vec(texts_cut, sg=sg, vector_size=size, window=window, min_count=min_count)
    return model_word2vec


def text2vec(texts_cut, model_word2vec, merge=True):
    """
    Convert text word sequences to word vector sequences.

    :param texts_cut: list of list of str, 文本的分词结果
    :param model_word2vec: gensim.models.word2vec.Word2Vec对象，训练好的Word2Vec模型
    :param merge: bool, 如果为True，则计算句子的平均向量，默认为True
    :return: numpy.ndarray, 句子向量的数组，如果merge为False，则返回词向量的列表
    """
    if texts_cut is None or model_word2vec is None:
        raise ValueError("texts_cut和model_word2vec不能为None.")

    texts_vec = []
    for text_cut in texts_cut:
        text_vec = [model_word2vec.wv[word] for word in text_cut if word in model_word2vec.wv]
        if not text_vec:
            text_vec = [np.zeros(model_word2vec.vector_size)]  # 处理空文本或者不在模型中的词
        texts_vec.append(text_vec)

    if merge:
        return np.array([np.mean(i, axis=0) for i in texts_vec])
    else:
        return texts_vec
