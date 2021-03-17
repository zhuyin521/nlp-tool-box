
# 词向量的调参技巧：
#
# 选择的训练word2vec的语料要和要使用词向量的任务相似，并且越大越好，论文中实验说明语料比训练词向量的模型更加的重要，所以要尽量收集大的且与任务相关的语料来训练词向量；
# 语料小（小于一亿词，约 500MB 的文本文件）的时候用 Skip-gram 模型，语料大的时候用 CBOW 模型；
# 设置迭代次数为三五十次，维度至少选 50，常见的词向量的维度为256、512以及处理非常大的词表的时候的1024维；

import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec

# 文件位置需要改为自己的存放路径
# 将文本分词
with open('./data/in_the_name_of_people.txt', encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open('./data/in_the_name_of_people_segment.txt', 'w', encoding="utf-8") as f2:
        f2.write(result)
# 加载语料
# 每一行对应一个句子（已经分词，以空格隔开），我们可以直接用LineSentence把txt文件转为所需要的格式。
sentences = word2vec.LineSentence('./data/in_the_name_of_people_segment.txt')
# 训练语料
path = get_tmpfile("word2vec.model")  # 创建临时文件
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=10, size=100)
model.save("./model/word2vec.model")
model = word2vec.Word2Vec.load("./model/word2vec.model")
# 输入与“贪污”相近的100个词
for key in model.wv.similar_by_word('贪污', topn=100):
    print(key)

# 获取指定词的词向量
print(model.wv.get_vector('今天'))