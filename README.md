<p align="center">
<!-- <img src="MGDCF_LOGO.png" width="400"/> -->
<!-- </p> -->

# Introduction

本项目是基于知识库与 Word2Vec 的多进程计算问答系统；
+ 通过使用 Docx 解析器将文档内容转换为文本数据，然后利用 jieba 进行文本分词
处理。接着，使用 gensim 库中的 Word2Vec 模型训练文本数据，生成词向量表示，用
于后续问题与文本相似度的计算。
+ 本问答系统的核心部分是 QARobot 类，其中包括训练模型和获取答案两个主要功
能。训练模型阶段将语料库中的文本转换为句向量，并使用多进程计算问题与文本的相
似度，以提高响应速度。在获取答案阶段，系统根据用户输入的问题，计算问题与知识
库中文本的相似度，并返回最相关的答案。整个系统通过 pickle 实现模型的持久化，使
得可以反复使用和更新模型，以提升问答系统的效率和准确性。

<p align="center">
<img src=".\architecture.png" height = "330" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall Framework of MGDCF.
</p>
 
# Requirements

+ Python 3.10.12
+ numpy == 1.26.2
+ jieba == 0.42.1
+ docx == 0.2.4
+ pandas == 2.1.4
+ gensim == 4.3.2
 


 
# Directory Structure

    ├── ReadMe.md            // 帮助文档
    
    ├── requirements.txt      // 环境依赖文件

    ├── QA_startup.py    // 主函数文件,启动机器人
    
    ├── utils            // tools
    
    │   └── __init__.py
    
    │   └──similarity.py    // 相似度计算
    
    │   └── text_segmentation.py      // 预处理分词
    
    │   └── text2vector.py   // 词向量训练

    │   ├── models
    
    │       └── __init__.py
    
    │       └── load_data.py      // 加载文本数据
    
    │       └── qa_system.py   // 多进程计算相似度的问答机器人
    
    │   └── __init__.py
    
 
# Run
+ 进行依赖项安装：pip install − r (chat_robot\) requirements.txt；
+ 注意：本项目需要以 QA_System 作为项目根路径打开，然后再安装依赖项，安装完成之后只需执行启动 python QA_startup.py 即可。
 

 
