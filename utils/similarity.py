import numpy as np
import pickle


def cal_similarity(v1, v2, similarity='cos'):
    """
    计算余弦相似度、欧几里得距离或杰卡德相似度

    :param v1: 第一个向量
    :param v2: 第二个向量
    :param similarity: 相似度方法, 'cos'、'Euclidean' 或 'Jaccard'
    :return: 相似度值
    """
    if similarity == 'cos':
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cos_sim
    elif similarity == 'Euclidean':
        euclidean_dist = np.linalg.norm(v2 - v1)
        return euclidean_dist
    elif similarity == 'Jaccard':
        intersection = np.logical_and(v1, v2).sum()
        union = np.logical_or(v1, v2).sum()
        jaccard_sim = intersection / union if union != 0 else 0.0  # handle division by zero
        return jaccard_sim
    else:
        raise ValueError('相似度方法应为 "cos"、"Euclidean" 或 "Jaccard"')


def cal_similarities(v, v_list, similarity='cos', modify=False):
    """
    计算向量与向量列表中每个向量的相似度

    :param v: array, 待比较的向量
    :param v_list: iter, 向量迭代器
    :param similarity: str, 'cos' 或 'Euclidean', 相似度计算方法
    :param modify: bool, 是否进行余弦修正
    :return: 相似度列表
    """

    def valid_vector(vec):
        return isinstance(vec, np.ndarray)

    def compute_similarity(vec1, vec2):
        if not valid_vector(vec1) or not valid_vector(vec2):
            return -999.0
        return cal_similarity(v1=vec1, v2=vec2, similarity=similarity)

    similarity_all = []
    if modify:
        valid_vectors = np.array([vec for vec in v_list if valid_vector(vec)])
        mean_vector = valid_vectors.mean(axis=0)
        v = v - mean_vector
        for vec in v_list:
            similarity_all.append(compute_similarity(v, vec - mean_vector))
    else:
        for vec in v_list:
            similarity_all.append(compute_similarity(v, vec))

    return similarity_all


def mul_cal_similarities(v, v_list, v_list_index, similarity, modify, path):
    similarity_all = cal_similarities(v, v_list, similarity, modify)
    with open(path, mode='wb') as f:
        pickle.dump([v_list_index, similarity_all], f)
