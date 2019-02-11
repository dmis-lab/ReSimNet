import pickle
import argparse


argparser = argparse.ArgumentParser()

argparser.add_argument('--file', type=str, default='')
args = argparser.parse_args()
print(args)


def main():
    dataset = pickle.load(open('./results/' + args.file, 'rb'))
    for key, value in dataset.items():
        print(key, value)
        break


if __name__ == '__main__':
    main()

    
