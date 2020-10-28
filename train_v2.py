import os
import json
from sklearn.metrics import classification_report

from model import BERTCustomModel, ENTITY_SEP_TOKEN


DATA_DIR = '/home/imyaboy888/cs598/hw3/bio-relation-extraction'

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
    train_json_path = os.path.join(DATA_DIR, 'data', '1.0alpha7.train.json')
    dev_json_path = os.path.join(DATA_DIR, 'data', '1.0alpha7.dev.json')

    print('[DEBUG] Run till here 01')

    X_train, y_train = prepare_data(train_json_path)
    X_dev, y_dev = prepare_data(dev_json_path)

    print('[DEBUG] Run till here 02')

    print("Number of train triples:", len(X_train))
    print("Number of train labels:", len(y_train))

    model = BERTCustomModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    print('[DEBUG] Run till here 03')

    print(classification_report(y_dev, predictions, digits=3))
    

if __name__ == "__main__":
    main()