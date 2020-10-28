import io
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from model import BERTCustomModel, ENTITY_SEP_TOKEN

def prepare_data(*paths):
    """
    Collect set of relations occurring in given samples.
    :param paths: Paths of json files containing samples.
    :return: Set of relation tuples.
    """
    # First, we read json samples to learn relations from
    samples = []
    for path in paths:
        with io.open(path, 'r', encoding='utf-8') as f:
            samples += json.load(f)

    # Collect all the occurring relations
    triples, labels = [], []
    lengths = []

    # Collect all the occurring relations
    for sample in samples:
        gold_relations = [tuple(x['participants']) for x in sample['interactions']]

        lengths.append(len(sample['text']))

        entities = sample['entities']
        num_entities = len(entities)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                for a in entities[i]['names']:
                    for b in entities[j]['names']:
                        triples.append((a + ENTITY_SEP_TOKEN + b, sample['text']))
                        triples.append((b + ENTITY_SEP_TOKEN + a, sample['text']))
                        if (i, j) in gold_relations:
                            labels.append(1)
                            labels.append(1)
                        else:
                            labels.append(0)
                            labels.append(0)

    print("Max number of characters:", max(lengths))
    print("Avg number of characters:", sum(lengths) / len(lengths))
    return triples, labels


def main():
    # Collect information on known relations
    self_path = os.path.realpath(__file__)
    self_dir = os.path.dirname(self_path)

    train_json_path = os.path.join(self_dir, 'data', '1.0alpha7.train.json')
    dev_json_path = os.path.join(self_dir, 'data', '1.0alpha7.dev.json')

    X_train, y_train = prepare_data(train_json_path)
    X_dev, y_dev = prepare_data(dev_json_path)

    print("Number of train triples:", len(X_train))
    print("Number of train labels:", len(y_train))

    model = BERTCustomModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    print(classification_report(y_dev, predictions, digits=3))
    

if __name__ == "__main__":
    main()