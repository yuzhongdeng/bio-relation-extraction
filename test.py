import os
import json
from collections import defaultdict
from train import prepare_data
from model import BERTCustomModel, ENTITY_SEP_TOKEN

DATA_DIR = '/Users/ydeng/projects/cs598/hw3/bio-relation-extraction'
#DATA_DIR = '/home/imyaboy888/cs598/hw3/bio-relation-extraction'

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


def collect_data(*paths):
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
    relations = set()
    for sample in samples:
        entities = sample['entities']
        for interaction in sample['interactions']:
            i, j = interaction['participants']
            for a in entities[i]['names']:
                for b in entities[j]['names']:
                    relations.add((a, b))
                    relations.add((b, a))

    return relations

    
def predict(sample, relations):
    text = sample['text']

    interactions = []
    sample['interactions'] = interactions

    entities = []
    sample['entities'] = entities
    entity_registry = dict()

    def find_mentions(entity):
        start = -1
        while True:
            start = text.find(entity, start + 1)
            if start < 0:
                break
            end = start + len(entity)
            yield start, end

    def register_entity(entity):
        if entity in entity_registry:
            return entity_registry[entity]

        idx = len(entities)
        mentions = list(find_mentions(entity))
        entities.append({
            'is_state': False,
            'label': 'protein',
            'names': {
                entity: {
                    'is_mentioned': True,
                    'mentions': mentions
                }
            },
            'is_mentioned': True,
            'is_mutant': False
        })
        entity_registry[entity] = idx
        return idx

    for a, b in relations:
        if a not in text or b not in text:
            continue
        # As the database is symmetric, omit duplicates
        if a >= b:
            continue

        # Ensure we have entity registered
        a_idx = register_entity(a)
        b_idx = register_entity(b)

        interactions.append({
            'participants': [a_idx, b_idx],
            'type': 'bind',
            'label': 1
        })

    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='Path to directory containing input.json.')
    parser.add_argument('output_dir', type=str,
                        help='Path to output directory to write predictions.json in.')
    parser.add_argument('shared_dir', type=str,
                        help='Path to shared directory.')
    args = parser.parse_args()

    # Collect information on known relations
    self_path = os.path.realpath(__file__)
    self_dir = os.path.dirname(self_path)

    # Read input samples and predict w.r.t. set of relations.
    model_path = os.path.join(self_dir, 'model')
    input_json_path = os.path.join(args.input_dir, 'input.json')
    output_json_path = os.path.join(args.output_dir, 'predictions.json')

    with io.open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X_test, y_test, i_test = prepare_data(input_json_path)
    model = BERTCustomModel(model_path)
    preds_test, fscore = model.test(X_test, y_test)
    predictions = process_preds(data, preds_test, i_test)
    
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=True)
    
if __name__ == "__main__":
    main()