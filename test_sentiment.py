from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

dataset = load_dataset('csv', data_files={'train': 'Corona_NLP_train.csv', 'test': 'Corona_NLP_test.csv'}, encoding = "ISO-8859-1")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def transform_labels(label):

    label = label['Sentiment']
    num = 0
    if label == 'Positive':
        num = 0
    elif label == 'Negative':
        num = 1
    elif label == 'Neutral':
        num = 2
    elif label == 'Extremely Positive':
        num = 3
    elif label == 'Extremely Negative':
        num = 4

    return {'labels': num}

def tokenize_data(example):
    return tokenizer(example['OriginalTweet'], padding='max_length')

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = dataset.map(tokenize_data, batched=True)

remove_columns = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'Sentiment']
dataset = dataset.map(transform_labels, remove_columns=remove_columns)





model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

train_dataset = dataset['train'].shuffle(seed=10).select(range(40000))
eval_dataset = dataset['train'].shuffle(seed=10).select(range(40000, 41000))

training_args = TrainingArguments("test", num_train_epochs=3, per_device_train_batch_size=4)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics,
)

trainer.train()




trainer.evaluate()