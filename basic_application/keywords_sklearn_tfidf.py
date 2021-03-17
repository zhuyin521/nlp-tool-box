import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 简体可以使用encoding = 'utf-8',繁体需要使用encoding='gbk'
stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='gbk').readlines()]
X, Y = ['\u4e00', '\u9fa5']
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"
# 用sklearn的tfidf提取关键词
# 把文章全部合并再计算tfidf再提取关键词语
# 此法提取一百个关键词的结果最终和jieba自带的分词结果重合度超过65% 速度能够大幅提高
tag = jieba.lcut(text.strip(), cut_all=False)
tag = [i for i in tag if len(i) >= 2 and X <= i <= Y and i not in stopwords]
tag_str = [' '.join(tag)]

vectorizer = CountVectorizer()
cif = vectorizer.fit_transform(tag_str)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(cif)
word = vectorizer.get_feature_names()  # 得到所有切词以后的去重结果列表
word = np.array(word)  # 把词语列表转化为array数组形式
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来
word_index = np.argsort(-weight)
word = word[word_index]  # 把word数组按照tfidf从大到小排序
tags = []
for i in range(100):
    tags.append(word[0][i])
