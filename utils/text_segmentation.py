import jieba

jieba.setLogLevel('WARN')


def texts_seg(texts=None, need_cut=True, word_len=1):
    """
    使用jieba进行文本分词

    :param texts: list of texts, 待分词的文本列表
    :param need_cut: bool, 是否需要进行分词，默认为True
    :param word_len: int, 最小词长，用于过滤短于指定长度的词，默认为1
    :return: 分词后的文本列表
    """
    if texts is None:
        return []

    if need_cut:
        if word_len > 1:
            texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
        else:
            texts_cut = [jieba.lcut(one_text) for one_text in texts]
    else:
        if word_len > 1:
            texts_cut = [[word for word in text if len(word) >= word_len] for text in texts]
        else:
            texts_cut = texts

    return texts_cut
