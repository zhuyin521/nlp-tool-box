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
sentences = word2vec.LineSentence('./data/in_the_name_of_people_segment.txt')
# 训练语料
path = get_tmpfile("word2vec.model")  # 创建临时文件
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=10, size=100)
model.save("./model/word2vec.model")
model = word2vec.Word2Vec.load("./model/word2vec.model")
# 输入与“贪污”相近的100个词
for key in model.wv.similar_by_word('贪污', topn=100):
    print(key)
