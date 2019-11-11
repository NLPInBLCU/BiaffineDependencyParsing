# BiaffineDependencyParsing
Various Encoder Layers (~~vanilla LSTM/Highway Droput LSTM/Transformer/~~BERTology) + Biaffine Decoder for Dependency Parsing
## Result

<table><tr><th rowspan="2">Model</th><th colspan="2">TEXT</th><th colspan="2">NEWS</th></tr><tr><td>test</td><td>dev</td><td>test</td><td>dev</td></tr><tr><td>AAAI2018</td><td>-</td><td>72.92</td><td>-</td><td>63.30</td></tr><tr><td>腾讯词向量+斯坦福模型</td><td>81.371</td><td>80.669</td><td>-</td><td>-</td></tr><tr><td>BERT 初始版本</td><td>81.01</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BERT+Transformer</td><td>82.35</td><td>-</td><td>-</td><td>-</td></tr></table>

## 相关项目：
- https://github.com/Hyperparticle/udify
- https://github.com/nikitakit/self-attentive-parser
- https://github.com/WangYuxuan93/CLBT
- https://github.com/stanfordnlp/stanfordnlp

## 使用
### 训练Train
```shell
python main.py -c config_files/bert_biaffine.yaml
```
### 验证Dev
```shell
python main.py -c config_files/bert_biaffine.yaml \
               --run dev --model_path <训练好的模型路径> \
               --input <测试输入conllu文件路径> \
               --output <测试输出conllu文件路径> \
               --use_cuda
```
### 推理Inference
```shell
python main.py -c config_files/bert_biaffine.yaml \
               --run inference --model_path <训练好的模型路径> \
               --input <输入conllu文件路径> \
               --output <输出conllu文件路径> \
               --use_cuda
```