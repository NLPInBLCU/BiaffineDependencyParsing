# -*- coding: utf-8 -*-
# Created by li huayong on 2020/4/21
import matplotlib.pyplot as plt
import mplcyberpunk
import pandas as pd


def load_data(file_path='metrics.csv'):
    df = pd.read_csv(file_path)
    data = {}
    for _, r in df.iterrows():
        data[r.model_name] = [r.text_las, r.news_las]
    return data


ys_data = load_data()

plt.style.use("cyberpunk")

plt.figure(figsize=(10, 5))
plt.title("Model Test Metrics in SemEval2016 Dataset", fontsize=18)
# plt.xticks(rotation=45)
plt.ylabel("LAS", fontsize=15)
# 设置刻度字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

xs = ['TextBook', 'News']
for y in ys_data:
    plt.plot(xs, ys_data[y], marker='o', label=y)
for ys in ys_data:
    for x, y in zip(xs, ys_data[ys]):
        plt.text(x, y, y, fontsize=10)

plt.legend(fontsize=14)
plt.grid(True)

mplcyberpunk.add_glow_effects()
plt.savefig('metrics_line.svg')
plt.show()

if __name__ == '__main__':
    pass
