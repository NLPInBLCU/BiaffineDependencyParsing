# TODO LIST
## Done
- [x] 跑通模型
- [x] 多卡
- [x] 跳过超过最大句长的句子
- [x] bert下简化的biaffine scorer
- [x] 删除无用、过时代码
- [x] 支持tensorboardX
- [x] BERT后添加一个self-attention层
- [x] 加入Layer Attention、Layer Dropout
- [x] 重构get optimizer
## High Priority
- [ ] dev数据集上调优
- [ ] 保存、加载
- [ ] 保存加载预处理的dataset
- [ ] 调整BERT 优化器的参数
- [ ] decoder部分的参数初始化
- [ ] 按照句长均分loss
- [ ] 按照累计句长划分batch
- [ ] Input Masking
- [ ] 多任务训练 text/news分成两个decoder一起训练（此时训练集也得分开）
- [ ] 多领域数据的采样（参考多语BERT）
- [ ] 多任务训练 + POS 标注
- [ ] 解决Biaffine分类（二分类、多分类）的类别不平衡问题
- [ ] 重构预测得到probs的后处理部分
- [ ] 支持roberta
## Low Priority
- [ ] 修改GraphVocab，支持过滤低频次的标签
- [ ] 支持xlnet
- [ ] 重构Transformer的输入
- [ ] 重构Transformer的encoder
- [ ] 重构CharRNN的输入
- [ ] 重构HLSTM的encoder
- [ ] 支持句法依存分析
- [ ] inverse square root learning rate decay (Vaswani et al., 2017). 
- [ ] 多任务训练 + NER
