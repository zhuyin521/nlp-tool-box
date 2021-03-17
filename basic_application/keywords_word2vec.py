# https://kexue.fm/archives/4316
# word2vec model 链接: https://pan.baidu.com/s/1ebvGiqmlr-TBtXp0-mo8JA  密码: ewsc
# 使用 word2vec 抽取关键词
# TF-IDF算法的效率是𝒪(N)，而用Word2Vec提取，效率显然是𝒪(N2)
# Word2Vec虽然仅仅开了窗口，但已经成功建立了相似词之间的联系，
# 也就是说，用Word2Vec做上述过程，事实上将“相似词语”进行叠加起来进行评估，相比之下，TF-IDF的方法，仅仅是将“相同词”叠加起来进行评估，
# 因此，我们说Word2Vec提取关键词，能够初步结合语义来判断了。而且，Word2Vec通过考虑p(wk|wi)来考虑了文章内部的关联，这里有点TextRank的味道了，是一个二元模型，而TF-IDF仅仅考虑词本身的信息量，仅仅是一个一元模型
# Word2Vec是基于神经网络训练的，自带平滑功能，哪怕两个词语在文本中未曾共现，也能得到一个较为合理的概率。
import numpy as np
import gensim
from collections import Counter
import pandas as pd
import jieba
# 可以根据自己的预料训练一个全新的模型
model = gensim.models.word2vec.Word2Vec.load('./model/word2vec.model')


# 此函数计算某词对于模型中各个词的转移概率p(wk|wi)
def predict_proba(oword, iword):
    # 获取输入词的词向量
    iword_vec = model[iword]
    # 获取保存权重的词的词库
    oword = model.wv.vocab[oword]
    oword_l = model.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
    return lprob


# 各个词对于某词wi转移概率的乘积即为p(content|wi)，
# 如果p(content|wi)越大就说明在出现wi这个词的条件下，此内容概率越大，
# 那么把所有词的p(content|wi)按照大小降序排列，越靠前的词就越重要，越应该看成是本文的关键词。


def keywords(s):
    s = [w for w in s if w in model]
    ws = {w: sum([predict_proba(u, w) for u in s]) for w in s}
    return Counter(ws).most_common()


s = u'太阳是一颗恒星'
print(pd.Series(keywords(jieba.cut(s))))

s = u'昌平区政府网站显示，明十三陵是世界上保存完整、埋葬皇帝最多的墓葬群，1961年被国务院公布为第一批全国重点文物保护单位，并于2003年被列为世界遗产名录。'
print(pd.Series(keywords(jieba.cut(s))))
