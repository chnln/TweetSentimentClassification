# Tweet情感分类

语义计算与知识检索 2022春季学期 第二次作业

[TOC]

# 1 实验背景

本次作业需要完成SemEval 2017 Task4的子任务A：推特情感分类。[数据集链接](http://alt.qcri.org/semeval2017/task4/index.php?id=results)

数据规模为：

|       | negative | neutral | positive | all   |
| ----- | -------- | ------- | -------- | ----- |
| train | 8528     | 24210   | 20822    | 53570 |
| test  | 3972     | 5937    | 2375     | 12284 |

其中训练集来自于2017年以前的SemEval中情感分类任务的语料，并增加了livejournal和sms两份语料文件；测试集为SemEval 2017 task4新提出的语料；同时，在模型训练时，本实验从训练集随机抽取1/10的量得到开发集。

# 2 实验环境

## 2.1 硬件与库依赖

- 本实验租用云GPU服务器平台[feature](featurize.cn)的实例，在RTX 3090显卡上运行。

- 本实验运行时的主要库环境为：

  | library       | version |
  | ------------- | ------- |
  | python        | 3.7.10  |
  | transformers  | 4.18.0  |
  | torch         | 1.10.0  |
  | cuda_toolkits | 11.3    |

## 2.2 文件结构

- `preprocess.ipynb`文件将原始`.tsv`和`.txt`转换成`.csv`文件，并定义了推特文本预处理函数。
- `bert.ipynb`和`bert_preprocessed.ipynb`文件为实验文件。
- `data`文件夹存放数据文件。
- `predictions`文件夹存放预测结果。
- `scores`文件夹存放评测结果。

# 3 实验内容

## 3.1 模型

![image-20220424232515023](/Users/nascent/Library/Application Support/typora-user-images/image-20220424232515023.png)

本实验在BERT（[Devlin et al., 2019](https://arxiv.org/abs/1810.04805)）模型上fine-tune来完成给定任务，具体在实验细节上，调用了Huggingface公司提供的[BERT序列分类模型](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification)和训练好的[bert-base-uncased](https://huggingface.co/bert-base-uncased) 模型。

```python
from transformers import Autotokenizer, AutoModelForSequenceClassification

sentiment2label = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
label2sentiment = {v:k for k, v in sentiment2label.items()}

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, label2id=sentiment2label, id2label=label2sentiment)
```

超参数设置如下：

| hyperparameter   | value |
| ---------------- | ----- |
| number of epochs | 3     |
| learning rate    | 5e-5  |
| batch size       | 16    |

## 3.2 评价指标

本文沿用SemEval 2017 task 4 Subtask A[论文](https://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4.pdf)中使用的指标，即以平均召回率AvgRec（即macro-recall）为主，准确率Accuracy和F1值F1-macro为辅。

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "AvgRec": recall_score(labels, predictions, average="macro"),
        "Accuracy": accuracy_score(labels, predictions),
        "F1-macro": f1_score(labels, predictions, average="macro")
    }
```

## 3.3 预处理

由于推特语料是网络语料，用语活泼且掺杂网络链接、用户名等，本实验意在比较预处理对推特情感分类任务效果的影响。实验组为将经过了预处理的文本送入BERT模型，对照组为将没有经过预处理的文本送入BERT模型。

具体预处理操作为：

```python
def preprocess_tweet(origin):
    tweet = re.sub(r"\\u2019", "'", origin).lower() # I\u2019m -> I'm
    tweet = re.sub(r"\\u002c", ",", tweet) # \u002c -> ,
    tweet = re.sub(r"@\S+", "@user", tweet)
    tweet = re.sub(r"https?:\/\/\S+", "http", tweet)
    tweet = re.sub(r"(www.)?[a-z]+\.(com)/?[a-z0-9]*", "http", tweet)
    return tweet
```

主要是考虑到训练集中的引号、逗号被错误转写为unicode字符，此外网址、用户名等较多。

# 4 实验结果

实验结果如下：

|        | AvgRec     | Accuracy   | F1-Score   |
| ------ | ---------- | ---------- | ---------- |
| 实验组 | **0.6960** | **0.6924** | **0.6898** |
| 对照组 | 0.6927     | 0.6854     | 0.6841     |

两组实验中，各类情感（negetive、neutral和positive）的得分情况详见`scores`文件夹，具体的预测结果见`predictions`文件夹。

# 5 分析与讨论

## 5.1 预处理的作用

可以看出，实验组的评测指标都略好于对照组，但优化效果有限。原因可能在于：

- 这次任务的测试集和训练集存在异质性。观察发现测试集中不存在引号和逗号被错误转写成unicode编码的情况，在这种情况下上文所做的预处理“拉近”了训练集和测试集的距离，可能是预处理产生作用的一个原因。
- 预处理还不够充分。网络语言更加活泼，比如“great”可能写成“grrrrrreat“，从这个角度出发可以对推特语料做更加细致的词形还原工作；此外，由于语料年份跨度大，一些俚语、流行语也会影响情感分类，可以引入相关词典或释义来帮助模型理解这些词汇。

## 5.2 模型的过拟合

| Step | Training Loss | Validation Loss | Avgrec     | Accuracy   | F1-macro   |
| ---- | ------------- | --------------- | ---------- | ---------- | ---------- |
| 500  | 0.7803        | 0.6593          | 0.7039     | 0.7051     | 0.6921     |
| 1000 | 0.6847        | 0.6171          | 0.6928     | 0.7286     | 0.7108     |
| 1500 | 0.6632        | 0.5566          | 0.7294     | 0.7627     | 0.7424     |
| 2000 | 0.656         | 0.5144          | 0.7636     | 0.7766     | 0.7664     |
| 2500 | 0.6391        | 0.5056          | 0.7952     | 0.7883     | 0.7817     |
| 3000 | 0.6337        | 0.4787          | 0.7646     | 0.7948     | 0.7795     |
| 3500 | 0.4551        | 0.4437          | 0.806      | 0.8176     | 0.8116     |
| 4000 | 0.4501        | 0.4093          | 0.8507     | 0.8363     | 0.8311     |
| 4500 | 0.4303        | 0.3578          | 0.848      | 0.8619     | 0.8566     |
| 5000 | 0.4447        | 0.3196          | 0.8802     | 0.8805     | 0.8775     |
| 5500 | 0.4332        | 0.2996          | 0.8871     | 0.8902     | 0.8876     |
| 6000 | 0.4212        | 0.2647          | 0.9101     | 0.9031     | 0.9028     |
| 6500 | 0.2417        | 0.2671          | 0.923      | 0.9141     | 0.9118     |
| 7000 | 0.2189        | 0.2412          | 0.9249     | 0.9259     | 0.9247     |
| 7500 | 0.2305        | 0.232           | 0.9362     | 0.9306     | 0.93       |
| 8000 | 0.2232        | 0.2             | 0.9383     | 0.9371     | 0.9353     |
| 8500 | 0.2206        | 0.1907          | 0.9444     | 0.9438     | 0.9435     |
| 9000 | 0.2048        | 0.1811          | **0.9471** | **0.9479** | **0.9474** |

从训练过程可以看出（如上表），模型在训练集上的loss和评测指标情况都非常好，甚至达到了90%以上的正确率、召回率和F1值，应该出现了过拟合的情况，说明训练过程还有很大的优化空间。

## 5.3 预训练模型 vs 线性回归模型

| epoch | train-acc | train-f1 | test-acc | test-f1  |
| ----- | --------- | -------- | -------- | -------- |
| 0     | 0.343633  | 0.235581 | 0.289593 | 0.160435 |
| 1     | 0.367158  | 0.265265 | 0.290045 | 0.172899 |
| 2     | 0.398057  | 0.293926 | 0.283258 | 0.184334 |
| 3     | 0.409059  | 0.305493 | 0.28371  | 0.184885 |
| 4     | 0.426264  | 0.320344 | 0.279638 | 0.185831 |
| 5     | 0.42919   | 0.322658 | 0.2819   | 0.185269 |
| 6     | 0.437851  | 0.330571 | 0.28371  | 0.185026 |
| 7     | 0.447214  | 0.338593 | 0.279186 | 0.18672  |
| 8     | 0.451896  | 0.340991 | 0.28009  | 0.186886 |
| 9     | 0.460323  | 0.349021 | 0.278281 | 0.18867  |

笔者在这个任务上复用了此前写的[线性分类模型](https://github.com/chnln/LinearModelForTextClassification)，但效果很差（如上图，评价指标为原任务采用的准确率、F1值）。原因可能在于：

- 在线性分类模型中，笔者使用的是词袋的特征表示法，在特征表示这方面相比BERT损失了大量信息。
- BERT的分类模型更加充分的捕捉了词、句和情感标签之间的关系。