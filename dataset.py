import pandas
import numpy
import json
import glob
import copy
from torch.utils.data import Dataset

from pytorch_pretrained_bert.tokenization import BertTokenizer


def partition(samples, n):
    '''
        divide samples into n partitions
    '''
    assert(len(samples) > n)  # there are more samples than partitions
    assert(n > 0)  # there is at least one partition
    size = len(samples) // n
    for i in range(0, len(samples), size):
        yield samples[i:i+size]

class EDUCollator(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __call__(self, batch):
        for sample in batch:
            # convert to index
            data = copy.deepcopy(sample)

            for i in range(0, len(data['data']), 512):
                done = False # to make sure all tokens have been converted to ids
                # boundary checking
                if i+512 > len(data['data']):
                    j = len(data['data'])
                else:
                    j = i+512
                while not done:
                    try:
                        data['data'][i:j] = self.tokenizer.convert_tokens_to_ids(
                            data['data'][i:j]) 
                        done = True
                    except KeyError as error:
                        err = error.args[0]
                        idx = data['data'].index(err)
                        data['data'][idx] = '[UNK]'

        # calculate the shape of batch
        all_sent_len = [len(data['data']) for sample in batch]
        longest_sent_sent = max(all_sent_len)
        data_size = len(batch)
        # prepare data (do padding)
        padded_data = numpy.ones((data_size, longest_sent_sent)) * 0
        input_mask = numpy.ones((data_size, longest_sent_sent)) * 0
        label = numpy.ones((data_size, longest_sent_sent)) * 0

        # copy over the actual sequences
        for idx, sample in enumerate(batch):

            padded_data[idx, 0:all_sent_len[idx]] = data['data']
            input_mask[idx, 0:all_sent_len[idx]] = [1] * len(data['data'])

            label[idx, 0:all_sent_len[idx]] = data['label']

        return padded_data, input_mask, label

class TransCollator(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __call__(self, batch):
        return batch[0]['bin_r_list'], batch[0]['gold_edu']

class AugCollator(object):
    def __init__(self, train_trans, train_rlat):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __call__(self, batch):
        for sample in batch:
            sample['sent1'] = self.tokenizer.tokenize(sample['sent1'])
            sample['sent2'] = self.tokenizer.tokenize(sample['sent2'])
            sample['aug_sent1'] = self.tokenizer.tokenize(sample['aug_sent1'])
            sample['aug_sent2'] = self.tokenizer.tokenize(sample['aug_sent2'])

            # while len(sample['sent1']) > 510:
            #     sample['sent1'].pop()
            # while len(sample['sent2']) > 510:
            #     sample['sent2'].pop()
            # if not self.train_rlat:
            #     while len(sample['sent3']) > 510:
            #         sample['sent3'].pop()

            sample['sent1'].insert(0, '[CLS]')
            sample['sent1'].append('[SEP]')
            sample['sent2'].insert(0, '[CLS]')
            sample['sent2'].append('[SEP]')
            sample['aug_sent1'].insert(0, '[CLS]')
            sample['aug_sent1'].append('[SEP]')
            sample['aug_sent2'].insert(0, '[CLS]')
            sample['aug_sent2'].append('[SEP]')

            # # convert to index
            # sample['sent1'] = self.tokenizer.convert_tokens_to_ids(
            #     sample['sent1'])
            # sample['sent2'] = self.tokenizer.convert_tokens_to_ids(
            #     sample['sent2'])
            # if not self.train_rlat:
            #     sample['sent3'] = self.tokenizer.convert_tokens_to_ids(
            #         sample['sent3'])

            # convert to index 
            for i in range(0, len(sample['sent1']), 512):
                # boundary checking
                if i+512 > len(sample['sent1']):
                    j = len(sample['sent1'])
                else:
                    j = i+512
                sample['sent1'][i:j] = self.tokenizer.convert_tokens_to_ids(
                            sample['sent1'][i:j]) 
            for i in range(0, len(sample['sent2']), 512):
                # boundary checking
                if i+512 > len(sample['sent2']):
                    j = len(sample['sent2'])
                else:
                    j = i+512
                sample['sent2'][i:j] = self.tokenizer.convert_tokens_to_ids(
                            sample['sent2'][i:j])
            for i in range(0, len(sample['aug_sent1']), 512):
                # boundary checking
                if i+512 > len(sample['aug_sent1']):
                    j = len(sample['aug_sent1'])
                else:
                    j = i+512
                sample['aug_sent1'][i:j] = self.tokenizer.convert_tokens_to_ids(
                            sample['aug_sent1'][i:j])
            for i in range(0, len(sample['aug_sent2']), 512):
                # boundary checking
                if i+512 > len(sample['aug_sent2']):
                    j = len(sample['aug_sent2'])
                else:
                    j = i+512
                sample['aug_sent2'][i:j] = self.tokenizer.convert_tokens_to_ids(
                            sample['aug_sent2'][i:j])

        # calculate the shape of batch (sent1,sent2,sent3)
        allsent1_len = [len(sample['sent1']) for sample in batch]
        allsent2_len = [len(sample['sent2']) for sample in batch]
        allaugsent1_len = [len(sample['aug_sent1']) for sample in batch]
        allaugsent2_len = [len(sample['aug_sent2']) for sample in batch]

        longestsent1_len = max(allsent1_len)
        longestsent2_len = max(allsent2_len)
        longestaugsent1_len = max(allaugsent1_len)
        longestaugsent2_len = max(allaugsent2_len)

        data_size = len(batch)

        # prepare data (do padding)
        # sample_id = numpy.ones((data_size, 1)) * 0
        padded_data1 = numpy.ones((data_size, longestsent1_len)) * 0
        padded_aug_data1 = numpy.ones((data_size, longestaugsent1_len)) * 0
        # input_mask1 = numpy.ones((data_size, longestsent1_sent)) * 0
        # token_type_ids = numpy.ones((data_size, longest_sent)) * 0
        padded_data2 = numpy.ones((data_size, longestsent2_len)) * 0
        padded_aug_data2 = numpy.ones((data_size, longestaugsent2_len)) * 0

        # input_mask2 = numpy.ones((data_size, longestsent2_sent)) * 0

        label = numpy.ones((data_size, 1)) * 0
        if self.train_rlat:
            center = numpy.ones((data_size, 1)) * 0   

        # copy over the actual sequences
        for idx, sample in enumerate(batch):
            # sample_id[idx, 0] = sample['id']
            sent1 = sample['sent1']
            sent2 = sample['sent2']
            aug_sent1 = sample['aug_sent1']
            aug_sent2 = sample['aug_sent2']

            padded_data1[idx, 0:allsent1_len[idx]] = sent1
            padded_data2[idx, 0:allsent2_len[idx]] = sent2
            padded_aug_data1[idx, 0:allaugsent1_len[idx]] = aug_sent1
            padded_aug_data2[idx, 0:allaugsent2_len[idx]] = aug_sent2

            # input_mask1[idx, 0:allSent1_len[idx]] = [1] * len(sent1)
            # input_mask2[idx, 0:allSent2_len[idx]] = [1] * len(sent2)
            # if not self.train_rlat:
                # input_mask3[idx, 0:allSent3_len[idx]] = [1] * len(sent3)
            # token_type_ids[idx, 0:all_len[idx]] = [
            #     0] * len(title1) + [1] * len(title2)
            label[idx, 0] = sample['label']
            if self.train_rlat:
                center[idx, 0] = sample['center']

        return padded_data1, padded_data2, label, center, \
               padded_aug_data1, padded_aug_data2

class RlatCollator(object):
    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __call__(self, batch):
        for sample in batch:

            sample['sent1'] = self.tokenizer.tokenize(sample['sent1'])
            sample['sent2'] = self.tokenizer.tokenize(sample['sent2'])
 
            while len(sample['sent1']) > 510:
                sample['sent1'].pop()
            while len(sample['sent2']) > 510:
                sample['sent2'].pop()

            sample['sent1'].insert(0, '[CLS]')
            sample['sent1'].append('[SEP]')
            sample['sent2'].insert(0, '[CLS]')
            sample['sent2'].append('[SEP]')

            # convert to index
            sample['sent1'] = self.tokenizer.convert_tokens_to_ids(
                sample['sent1'])
            sample['sent2'] = self.tokenizer.convert_tokens_to_ids(
                sample['sent2'])

        # calculate the shape of batch (sent1,sent2,sent3)
        allSent1_len = [len(sample['sent1']) for sample in batch]
        allSent2_len = [len(sample['sent2']) for sample in batch]

        longestSent1_sent = max(allSent1_len)
        longestSent2_sent = max(allSent2_len)

        data_size = len(batch)

        # prepare data (do padding)
        # sample_id = numpy.ones((data_size, 1)) * 0
        padded_data1 = numpy.ones((data_size, longestSent1_sent)) * 0
        input_mask1 = numpy.ones((data_size, longestSent1_sent)) * 0
        # token_type_ids = numpy.ones((data_size, longest_sent)) * 0
        padded_data2 = numpy.ones((data_size, longestSent2_sent)) * 0
        input_mask2 = numpy.ones((data_size, longestSent2_sent)) * 0

        label = numpy.ones((data_size, 1)) * 0
        center = numpy.ones((data_size, 1)) * 0   
        sub_label = numpy.ones((data_size, 1)) * 0

        # copy over the actual sequences
        for idx, sample in enumerate(batch):
            # sample_id[idx, 0] = sample['id']
            sent1 = sample['sent1']
            sent2 = sample['sent2']

            padded_data1[idx, 0:allSent1_len[idx]] = sent1
            padded_data2[idx, 0:allSent2_len[idx]] = sent2

            input_mask1[idx, 0:allSent1_len[idx]] = [1] * len(sent1)
            input_mask2[idx, 0:allSent2_len[idx]] = [1] * len(sent2)

            # token_type_ids[idx, 0:all_len[idx]] = [
            #     0] * len(title1) + [1] * len(title2)
            label[idx, 0] = sample['label']
            center[idx, 0] = sample['center']
            sub_label[idx, 0] = sample['sub_label']

        return padded_data1, padded_data2, label, center, sub_label

class EDUDataset(Dataset):
    def __init__(self, data: list, label_map: dict):
        # required to map the labels to integer values
        self.data = data
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = []
        for i in self.data[idx][1]:
            label.append(self.label_map[i])
        sample = {
            'data': self.data[idx][0],
            'label': label,
        }
        return sample

class TransDataset(Dataset):
    '''
        dataset for training structure with dynamic oracle
    '''
    def __init__(self, data: list):
        # required to map the labels to integer values
        self.data = data
        # self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = {
            'bin_r_list': self.data[idx][0],
            'gold_edu': self.data[idx][1],
        }
        return sample

class AugDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # we wont use label & center so set to 0
        sample = {
            'sent1': self.data[idx][0],
            'sent2': self.data[idx][1],
            'label': 0,
            'center': 0,
            'aug_sent1': self.data[idx][3],
            'aug_sent2': self.data[idx][4], 
        }

        return sample

class RlatDataset(Dataset):
    def __init__(self, data: list, label_map: dict, center_map: dict, sub_label_map: dict):
        # required to map the labels to integer values
        self.data = data
        self.label_map = label_map
        self.sub_label_map = sub_label_map
        self.center_map = center_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'sent1': self.data[idx][0],
            'sent2': self.data[idx][1],
            'label': self.label_map[self.data[idx][2].split('_')[0]],
            'center': self.center_map[self.data[idx][2].split('_')[1]],
            'sub_label': self.sub_label_map[self.data[idx][3]],
        }

        return sample
