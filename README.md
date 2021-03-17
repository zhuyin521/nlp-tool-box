# nlp-tool-box
提供nlp常用的基础工具箱,开箱即用,方便快速进行开发和想法验证


## 基础组件
- [ ] 分词
    - [ ] jieba
    - [ ] 新词发现 - 无监督构建词库 
- [ ] 文本表示
    - [ ] word2vec

## 底层应用
- [ ] 文本纠错
- [ ] 相似度计算
- [ ] 命名实体识别
- [ ] 文本分类聚类
- [ ] 搜索
- [ ] 关键词抽取
    - [ ] word2vec 



## docker 
docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
docker run --name face-d1 --gpus all -v /Users/zhuyin/:/home/work/ -itd nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

--不带gpu Force  docker run --name face-d1  -v /Users/zhuyin/:/home/work/ -itd nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04   
docker exec -it face-d1 /bin/bash
