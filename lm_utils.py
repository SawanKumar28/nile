import pandas as pd
import os
import pickle
import torch

from torch.utils.data import DataLoader, Dataset

EXP_TOKEN = '[EXP]'
EOS_TOKEN = '[EOS]'
class TSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False):
        self.print_count = 5
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        if os.path.exists(cached_features_file):
            print ('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            text1 = '{} {} '.format(r['input'], EXP_TOKEN)
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['target']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.eos_token_id] * (block_size-len(tokenized_text))
            if self.print_count > 0:
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
            return (tokenized_text, prompt_length, total_length)

        self.examples = data.apply(create_example, axis=1).to_list()
        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2]

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size):
        assert os.path.isfile(file_path)
        data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        print (data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        self.data.to_csv(filename, sep='\t')
