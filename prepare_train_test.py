import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="snli") #snli, mnli
    parser.add_argument("--create_data", action="store_true")
    parser.add_argument("--filter_repetitions", action="store_true")

    #For merging
    parser.add_argument("--merge_data", action="store_true")
    parser.add_argument("--merge_single", action="store_true")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input_prefix", type=str, default="dummy")

    #for shuffled evaluation
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()
    tqdm.pandas()

    args.dataset = args.dataset.lower()

    s1,s2 = 'sentence1', 'sentence2'
    index_col = 'pairID'
    gold_label = 'gold_label'
    data_labels = ['entailment', 'neutral', 'contradiction']
    if args.dataset == 'snli':
        input_path = './external/esnli/dataset'
        filenames = {
            'dev': 'esnli_dev.csv',
            'train': 'esnli_train.csv',
            'test': 'esnli_test.csv'
        }
        sep = ','
        data_index_col = 'pairID'
        data_gold_label = 'gold_label'
        quotechar = '"'
        quoting = 0
        data_s1,data_s2 = 'Sentence1', 'Sentence2'
        label_map = None
        skip_segregation = False
        explanation_available = True
        e1 = 'Explanation_1'
        output_root = './dataset_snli'
    elif args.dataset == 'mnli':
        input_path = './external/MNLI'
        filenames = {
            'train': 'train.tsv',
            'dev': 'dev_matched.tsv',
            'dev_mm': 'dev_mismatched.tsv'
        }
        sep = '\t'
        data_index_col = 'pairID'
        data_gold_label = 'gold_label'
        quotechar = None
        quoting = 3
        data_s1,data_s2 = 'sentence1', 'sentence2'
        label_map = None
        skip_segregation = True
        explanation_available = False
        output_root = './dataset_mnli'
    else:
        raise ValueError("dataset not supported")

    if args.create_data:
        data = {}
        for split in filenames:
            data[split] = pd.read_csv(os.path.join(input_path, filenames[split]),
                            index_col=data_index_col, sep=sep, quotechar=quotechar, quoting=quoting)
            data[split] = data[split].rename(columns={data_s1:s1, data_s2:s2, data_gold_label:gold_label})
            data[split].index.name = index_col
            if label_map: data[split][gold_label] = data[split][gold_label].apply(label_map)
                
            print ('\n Split {} Len {}'.format(split, len(data[split])))
            print (data[split][gold_label].value_counts())

            if args.filter_repetitions and split == "train" and args.dataset == 'snli':
                print ("Filtering repetitions")
                def has_repetition(r):
                    exp = r[e1].lower()
                    p = r[s1].lower()
                    h = r[s2].lower()
                    return True if p in exp or h in exp else False
                cond = data[split].apply(has_repetition, axis=1)
                print ("#cases with repetitions:", cond.sum())
                data[split] = data[split][cond==False]
                print ('Updated Split {} Len {}'.format(split, len(data[split])))
                print (data[split][gold_label].value_counts())
                cond = data[split].apply(has_repetition, axis=1)
                print ("#cases with repetitions:", cond.sum())

            label = "all"
            examples = data[split]
            dpath = os.path.join(output_root, label)
            os.makedirs(dpath) if not os.path.exists(dpath) else None
            fname = os.path.join(dpath, '{}_data.csv'.format(split))
            examples.to_csv(fname)

        if not skip_segregation:
            for label in data_labels:
                for split in filenames:
                    dpath = os.path.join(output_root, label)
                    os.makedirs(dpath) if not os.path.exists(dpath) else None
                    fname = os.path.join(dpath, '{}_data.csv'.format(split))
                    examples = data[split][data[split][gold_label] == label]
                    print ('Saving {} | {} | {} examples'.format(label, split, len(examples)))
                    examples.to_csv(fname)

    def generate_prompt(r):
        inp = 'Premise: {} Hypothesis: {}'.format(r[s1], r[s2])
        return inp

    if args.create_data:
        labels = ["all"]
        if not skip_segregation: labels.extend(data_labels)

        for label in labels:
            dpath =  os.path.join(output_root, label)
            for split in filenames:
                fname = os.path.join(dpath, '{}_data.csv'.format(split))
                examples = pd.read_csv(fname, index_col=index_col)
                print ('Processing {} | {} | {} examples'.format(label, split, len(examples)))
                print (examples[gold_label].value_counts())
                examples['input'] = examples[[s1, s2]].progress_apply(generate_prompt, axis=1)
                columns_to_write = ['input']
                if explanation_available:
                    examples['target'] = examples[e1]
                    columns_to_write.append('target')
                print ('Writing')
                fname = os.path.join(dpath, '{}.tsv'.format(split))
                examples[columns_to_write].to_csv(fname, sep='\t')
   
    if args.merge_data:
        if args.merge_single:
            suffixes = ["all"]
            output_suffix = "_all"
        else:
            suffixes = ["entailment", "contradiction", "neutral"]
            output_suffix = ""
        split = args.split
        fname_csv = os.path.join(output_root, 'all', '{}_data.csv'.format(split))
        d_csv = pd.read_csv(fname_csv, index_col=index_col)

        for s in suffixes:
            fname_tsv = os.path.join(output_root, 'all', '{}{}_{}.tsv'.format(args.input_prefix, s, split))
            d_tsv = pd.read_csv(fname_tsv, index_col=index_col, sep='\t')

            d_csv['{}_explanation'.format(s)] = d_tsv['Generated_Explanation']

        
        print (d_csv.head(5))
        fname = os.path.join(output_root, 'all', '{}{}_{}{}.csv'.format(args.input_prefix, split, "merged", output_suffix))
        d_csv.to_csv(fname)

    if args.shuffle:
        split = args.split
        fname = os.path.join(output_root, 'all', '{}{}_{}.csv'.format(args.input_prefix, split, "merged"))
        d_csv = pd.read_csv(fname, index_col=index_col)

        d_csv_shuffled = d_csv.copy()
        for l in ["entailment", "contradiction", "neutral"]:
            d_csv_shuffled['{}_explanation'.format(l)] = np.random.choice(
                            d_csv_shuffled['{}_explanation'.format(l)].values,
                            len(d_csv_shuffled), replace=False)
        fname_csv_out = os.path.join(output_root, 'all', '{}shuffled{}_{}.csv'.format(args.input_prefix, split, "merged"))     
        d_csv_shuffled.to_csv(fname_csv_out)

        
