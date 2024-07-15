import pickle
import warnings
import pandas as pd
from multiprocessing import Process
from chat_robot.utils.text_segmentation import texts_seg
from chat_robot.utils.text2vector import text2vec, train_dict
from chat_robot.utils.similarity import mul_cal_similarities

warnings.filterwarnings('ignore')


class QARobot:
    def __init__(self):
        self.mode = None
        self.model_word2vec = None
        self.texts_vector = None
        self.texts_all = []

    def train(self, texts, reuse=True, mode='knowledge'):
        """
        训练语料库得到句向量
        :param texts: list, 语料库列表
        :param reuse: bool, 是否增加语料库，True添加，False覆盖
        :param mode: str, 回答的类型，'knowledge' or 'chat'
        :return: None
        """
        word_len = 2
        self.mode = mode

        # 添加到原有知识列表
        if reuse:
            self.texts_all += texts
        else:
            self.texts_all = texts

        # 分词处理
        texts_cut_ = texts_seg(texts=self.texts_all, need_cut=True, word_len=word_len)

        # 词向量模型训练
        self.model_word2vec = train_dict(texts_cut=texts_cut_, sg=0, size=128, window=5, min_count=1)

        # 计算句向量
        self.texts_vector = text2vec(texts_cut=texts_cut_, model_word2vec=self.model_word2vec, merge=True)

    def get_answer(self, question='', sample=50000, similarity='cos', modify=False, threshold=0.0, num_answer=2,
                   process_num=2):
        """
        根据问题找到相似内容
        :param question: str, 问题
        :param sample: int, 答案抽样数量，避免计算过慢
        :param similarity: str, 'cos' 或 'Euclidean'，相似度计算方法
        :param modify: bool, 是否进行余弦修正
        :param threshold: float, 相似度阈值
        :param num_answer: int, 返回的答案数
        :param process_num: int, 进程数
        :return: list or str, 答案列表或者字符串
        """
        mode = self.mode
        texts_vec = self.texts_vector
        texts_all = self.texts_all

        word_len = 2    # 知识库模式下的分词最小长度

        # 对问题进行分词处理
        ask_cut = texts_seg(texts=[question], need_cut=True, word_len=word_len)

        # 计算问题的向量表示
        ask_vec = text2vec(texts_cut=ask_cut, model_word2vec=self.model_word2vec, merge=True)

        # 多进程计算相似度
        index_start = 0
        process_list = []

        for i in range(process_num):
            len_n = len(texts_vec[index_start:(index_start + sample)])
            texts_vec_part = texts_vec[index_start:(index_start + sample)]
            v_list_index = list(range(index_start, index_start + min(len_n, sample)))

            po = Process(target=mul_cal_similarities, kwargs={
                'v': ask_vec[0],
                'v_list': texts_vec_part,
                'v_list_index': v_list_index,
                'similarity': similarity,
                'modify': modify,
                'path': './' + str(i) + '.pkl'
            })

            process_list.append(po)
            index_start += sample

            if index_start > len(texts_vec):
                process_num = i + 1
                break

        # 启动子进程
        for process in process_list:
            process.start()

        # 等待子进程全部结束
        for process in process_list:
            process.join()

        texts_index = []
        similarity_all = []

        for i in range(process_num):
            with open('./' + str(i) + '.pkl', mode='rb') as f:
                r = pickle.load(f)
                texts_index += r[0]
                similarity_all += r[1]

        text_similarity = pd.DataFrame({'texts_index': texts_index, 'similarity': similarity_all},
                                       columns=['texts_index', 'similarity'])
        text_similarity_sort = text_similarity.sort_values(by='similarity', ascending=False)

        ask_similarity_index = list(
            text_similarity_sort.loc[text_similarity_sort['similarity'] >= threshold, 'texts_index'])[:num_answer]

        if not ask_similarity_index:
            return ['没有找到匹配的内容']
        else:
            return [texts_all[i] for i in ask_similarity_index]
