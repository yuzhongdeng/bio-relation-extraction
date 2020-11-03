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
    model = BERTCustomModel()
    preds_test = model.test(X_test, y_test)
    predictions = process_preds(data, preds_test, i_test)
    
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=True)
    
if __name__ == "__main__":
    main()