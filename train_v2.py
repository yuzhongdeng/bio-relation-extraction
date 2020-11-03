import os
import json
from model import BERTCustomModel, ENTITY_SEP_TOKEN

DATA_DIR = '/Users/ydeng/projects/cs598/hw3/bio-relation-extraction'
#DATA_DIR = '/home/imyaboy888/cs598/hw3/bio-relation-extraction'

def prepare_data(*paths):
    # First, we read json samples to learn relations from
    samples = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            samples += json.load(f)

    # Collect all the occurring relations
    triples, labels, indices = [], [], []

    # Collect all the occurring relations
    for idx, sample in enumerate(samples):
        gold_relations = [tuple(x['participants']) for x in sample['interactions']]
        text = sample['text']

        entities = sample['entities']
        num_entities = len(entities)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                for a in entities[i]['names']:
                    for b in entities[j]['names']:
                        triples.append((f'{a} {ENTITY_SEP_TOKEN} {b}', text))
                        triples.append((f'{b} {ENTITY_SEP_TOKEN} {a}', text))
                        if (i, j) in gold_relations:
                            labels.append(1)
                            labels.append(1)
                        else:
                            labels.append(0)
                            labels.append(0)
                        indices.append((idx, i, j))
                        indices.append((idx, i, j))

    print(f"Number of triples: {len(triples)}, labels: {len(labels)}, indices: {len(indices)}")
    return triples, labels, indices


def process_preds(samples, pairwise_preds, indices):
    interactions = defaultdict(set)
    for i, (sample_idx, a_idx, b_idx) in enumerate(indices):
        if pairwise_preds[i]:
            interactions[sample_idx].add((a_idx, b_idx))

    predictions = []
    for sample_idx, sample in enumerate(samples):
        sample = sample.copy()
        sample['interactions'] = []
        for a_idx, b_idx in interactions[sample_idx]:
            sample['interactions'].append({
                'participants': [a_idx, b_idx],
                'type': 'bind',
                'label': 1
            })
        predictions.append(sample)

    return predictions

def main():

    # Collect information on known relations
    train_json_path = os.path.join(DATA_DIR, 'data', '1.0alpha7.train.json')
    dev_json_path = os.path.join(DATA_DIR, 'data', '1.0alpha7.dev.json')
    model_out_path = os.path.join(DATA_DIR, 'model')

    X_train, y_train, i_train = prepare_data(train_json_path)
    X_dev, y_dev, i_dev = prepare_data(dev_json_path)

    model = BERTCustomModel()
    model.train(X_train, y_train, X_dev, y_dev)
    model.save(model_out_path)
    

if __name__ == "__main__":
    main()