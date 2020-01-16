# 分布式训练

## 测试环境

### 服务器：

单机；双卡 V100-16GB

### 参数：

train_batch_size: 40

dev_batch_size: 20

max_epochs: 3

## 使用Distributed Data Parallel

### 训练时长

570秒

### 精度

LAS: 0.79389

UAS: 0.88187

### 显存占用

卡1: 13461M

卡2: 13465M

## 使用Data Parallel

### 训练时长

565秒

### 精度

LAS: 0.79535

UAS: 0.88226

### 显存占用

卡1: 14221M

卡2: 11441M

## 单卡

### 训练时长

833秒

### 精度

LAS: 0.804645

UAS: 0.891561

### 显存占用

卡1: 12771M