import sys
import pickle
import warnings
from model.qa_system import QARobot
from model.load_data import load_data

warnings.filterwarnings('ignore')
sys.path.append("..")


def main():
    # 加载模型
    with open('./QA.pkl', mode='rb') as f_:
        chat_model = pickle.load(f_)

    print('您好，我是基于知识库与Word2Vec的多进程计算问答机器人小鸭！'
          '\n欢迎向我提问有关您的本地知识库的相关内容!(退出请键入exit)\n\n')
    while True:
        print('Question:')
        ask = input()
        if ask == 'exit':
            break
        else:
            answer = chat_model.get_answer(question=ask, sample=500, similarity='cos',
                                           modify=False, threshold=0, num_answer=3, process_num=2)
            # 若问题比较宽泛，可选择输入多个最终答案
            # print('In decreasing order of score:')
            # for n, i in enumerate(answer):
            #     print('Answer%d: %s' % (n + 1, i))

            print('Answer: %s' % answer[0])
            print()


if __name__ == '__main__':
    # 训练模型并保存
    texts = load_data()
    chat_robot = QARobot()
    chat_robot.train(texts=texts[:], mode='knowledge')
    with open('./QA.pkl', mode='wb') as f:
        pickle.dump(chat_robot, f)

    # 运行问答机器人
    main()
