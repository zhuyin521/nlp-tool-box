import jieba
import paddle
import jieba.posseg as pseg

# DAG分词  使用词典，此词典记录每个词的词频，词典由语料库中统计得到
# 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)


paddle.enable_static()
jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
jieba.enable_parallel(4)  # 启动多进程 提高分词性能
strs = ["我来到北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]
# 使用paddle模式  利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。同时支持词性标注
for str in strs:
    seg_list = jieba.cut(str, use_paddle=True)
    print("Paddle Mode: " + '/'.join(list(seg_list)))

# 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# 精确模式，试图将句子最精确地切开，适合文本分析；
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

# 载入词典
jieba.load_userdict("jieba_dict.txt")

jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
jieba.del_word('自定义词')

test_sent = (
    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
words = jieba.cut(test_sent)
print('/'.join(words))

print("=" * 40)
#  结合词性 进行标注

result = pseg.cut(test_sent)

for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')

# 用于新词发现


#  基于CRF分词
from pyhanlp import *

string = '我喜欢北京冬奥会'

HanLP.Config.ShowTermNature = False

print(HanLP.segment(string))
