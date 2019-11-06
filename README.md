# BiaffineDependencyParsing
Various Encoder Layers (~~vanilla LSTM/Highway Droput LSTM/Transformer/~~BERTology) + Biaffine Decoder for Dependency Parsing
## Result

<table><tr><th rowspan="2">Model</th><th colspan="2">TEXT</th><th colspan="2">NEWS</th></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>NLPCC</td><td></td><td>80.40</td><td></td><td>69.66</td></tr><tr><td>腾讯词向量+斯坦福模型</td><td></td><td>81.05</td><td></td><td></td></tr><tr><td>BERT 初始版本</td><td>81.01</td><td></td><td></td><td></td></tr><tr><td>BERT+Transformer</td><td>82.35</td><td></td><td></td><td></td></tr></table>

## 相关项目：
- https://github.com/Hyperparticle/udify
- https://github.com/nikitakit/self-attentive-parser
- https://github.com/WangYuxuan93/CLBT
- https://github.com/stanfordnlp/stanfordnlp

