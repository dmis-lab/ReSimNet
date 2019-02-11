import pickle
import argparse
from tasks.drug_task import DrugDataset

argparser = argparse.ArgumentParser()

argparser.add_argument('--file-path', type=str, 
                    default='tasks/data/drug/drug(v0.5).pkl')
argparser.add_argument('--save-path', type=str,
                    default='results')
args = argparser.parse_args()

def main():
    pair = {}
    
    dataset_l = pickle.load(open(args.file_path, 'rb'))
    dataset = dataset_l.dataset
    k_set = dataset_l.known
    test_data = dataset['te']
    for idx, item in enumerate(test_data):
        d1 = item[0]
        d2 = item[1]
        ds = (d1, d2)
        if d1 in k_set and d2 in k_set:
            label = 'KK'
        elif d1 not in k_set and d2 not in k_set:
            label = 'UU'
        else:
            label = 'KU'
        pair[ds] = label

    pickle.dump(pair, open('{}/testset.pkl'.format(
                args.save_path), 'wb'))

if __name__ == '__main__':
    main()
