import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data

from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

import random
import os
import csv
import copy
import sys
import glob
from collections import deque
from tqdm import tqdm
from preprocessor import Preprocessor
from dataset import TransCollator, TransDataset

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

os.makedirs("oracle_trans/mix_variation/", exist_ok=True)

tag_to_ix = {"shift":0, "reduce":1 }

def convert_to_bert(sent, tokenizer):
    sent = tokenizer.tokenize(sent)
    sent.insert(0, '[CLS]')
    sent.append('[SEP]')
    for i in range(0, len(sent), 512):
        # boundary checking
        if i+512 > len(sent):
            j = len(sent)
        else:
            j = i+512
        sent[i:j] = tokenizer.convert_tokens_to_ids(sent[i:j]) 
    return torch.tensor(sent, dtype=torch.long).cuda()

class NetTrans(nn.Module):

    def __init__(self, embedding_dim, tagset_size, batch_size):
        super(NetTrans,self).__init__()

        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        # BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        # freeze
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(self.embedding_dim*3, self.tagset_size)
        # self.hidden2tag = nn.Linear(embedding_dim*3, 512)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids1, input_ids2, input_ids3, aug_flag=None, input_mask1=None, input_mask2=None, input_mask3=None, token_type_ids=None, labels=None):
        '''
            Args:
                input_ids: [batch_size, seq_length]
                aug_flag: 
                    according to the requirement of `Kullback-Leibler divergence` Loss
                    the `input` given is expected to contain *log-probabilities* and is not restricted to a 2D Tensor.
                    The targets are given as *probabilities* (i.e. without taking the logarithm).

        '''

        # use sliding window approach to deal with BERT length 512 restriction
        # use narrow(dimension, start, length) to slice tensor 
        out_list = []
        for input_ids in [input_ids1, input_ids2, input_ids3]:
            input_ids_list = []
            attention_mask_list = []
            if input_ids.size()[1] > 512:
                for i in range(0, input_ids.size()[1], 256): # step size : 256
                    step = 512 if (i+512 <= input_ids.size()[1]) else input_ids.size()[1]-i
                    input_ids_list.append(input_ids.narrow(1, i, step))
                    # attention_mask_list.append(attention_mask.narrow(1, i, step))
                # send to BERT sequentially
                sequence_output_list = []
                for idx in range(0, len(input_ids_list)):         
                    # sequence_output, _ = self.bert(input_ids_list[idx], attention_mask_list[idx], output_all_encoded_layers=False)
                    sequence_output, _ = self.bert(input_ids_list[idx], output_all_encoded_layers=False)
                    sequence_output = self.dropout(sequence_output)
                    sequence_output_list.append(sequence_output)
                # combine by average the overlapping part
                sequence_output = []
                for i in range(0, len(sequence_output_list)-1):
                    if i == 0:
                        sequence_output.append(sequence_output_list[i][:, :256, :])
                    sequence_output.append((sequence_output_list[i][:, 256:, :] + sequence_output_list[i+1][:, :256, :]) /2)
                sequence_output = torch.cat(sequence_output, 1)

            else: 
                sequence_output, _ = self.bert(input_ids, output_all_encoded_layers=False)
                sequence_output = self.dropout(sequence_output) 

            out_list.append(sequence_output)

        pooled_list = []
        for out in out_list:
            pooled_list.append(out.max(1)[0])

        pooled = torch.cat(pooled_list, 1)
        # after combination, use a linear layer to reduce hidden dimension to tagset size
        logits = self.hidden2tag(pooled)
        return logits

class ModelTrans():
    def __init__(self, train_data, test_data, valid_data, 
                 embedding_dim, tagset_size, batch_size):

        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data
        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size = tagset_size # relation label size
        self.batch_size = batch_size

        self.model = NetTrans(self.embedding_dim, self.tagset_size, self.batch_size).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr= 5e-5)
        self.criterion = nn.CrossEntropyLoss().cuda() # delete ignore_index = 0 

    def train(self):

        collate_fn = TransCollator()

        train = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )
        valid = DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )
        test = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

        self.model.zero_grad()

        for epoch in range(30): 
            running_loss = 0.0
            step = 0
            self.model.train()

            trange = tqdm(enumerate(train),
                          total=len(train),
                          desc='modelTrans train',
                          ascii = True)

            for i, (bin_r_list, gold_edu) in trange:
                running_loss, step = self.train_paragraph(trange, bin_r_list, gold_edu, running_loss, step)

            # with open('{}'.format('oracle_trans/mix_variation/mix_variation.csv'), 'w') as f:  
            #     f.write('epoch,train,test\n')

            print("\n")
            print('[%d] loss: %.5f' %
                  (epoch + 1, running_loss*self.batch_size / len(train)))

            with torch.no_grad():
                train_acc = self.test_accuracy("train", self.model, train)
            with torch.no_grad():
                valid_acc = self.test_accuracy("train", self.model, valid)
            with torch.no_grad():
                test_acc = self.test_accuracy("test", self.model, test)

                # with open('{}'.format('oracle_trans/mix_variation/mix_variation.csv'), 'a') as f:
                #     writer = csv.writer(f, delimiter=',')
                #     writer.writerow(
                #         [epoch+1, train_acc, test_acc])

                # torch.save(
                #     model.state_dict(),
                #     'oracle_trans/mix_variation/model_trans.pkl.{}'.format(epoch+1))

    def train_paragraph(self, trange, bin_rlat_list, golden_edu, running_loss, step):
        alpha = 0.7
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # teacher mode
        data = copy.deepcopy(bin_rlat_list)

        stack = []
        queue = deque(golden_edu)
        while data != []:
            if len(stack) < 2: 
                stack.append(queue.popleft())
            else:
                if(queue != deque([])):
                    step += 1
                    self.model.zero_grad() 
                    self.optimizer.zero_grad()
                    relation = stack[len(stack)-2] + "|" + stack[len(stack)-1]
                    du = stack[len(stack)-2] + stack[len(stack)-1]

                    if relation in data: # reduce

                        sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                        sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                        sent3_torch = convert_to_bert(queue[0], tokenizer)
                        label_torch = torch.tensor([tag_to_ix['reduce']], dtype=torch.long).cuda()

                        score = self.model(
                            sent1_torch.view(1,-1),
                            sent2_torch.view(1,-1),
                            sent3_torch.view(1,-1),
                        )
                        loss = self.criterion(
                            score.view(1, len(tag_to_ix)),
                            label_torch.view(1)
                        )
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item() 

                        trange.set_postfix(
                            {'loss' : '{0:1.5f}'.format(running_loss / (step + 1))}
                        )
                        stack.pop()
                        stack.pop()
                        stack.append(du)
                        data.remove(relation)
                    else: # shift
                        sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                        sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                        sent3_torch = convert_to_bert(queue[0], tokenizer)
                        label_torch = torch.tensor([tag_to_ix['shift']], dtype=torch.long).cuda()

                        score = self.model(
                            sent1_torch.view(1,-1),
                            sent2_torch.view(1,-1),
                            sent3_torch.view(1,-1),
                        )
                        loss = self.criterion(
                            score.view(1, len(tag_to_ix)),
                            label_torch.view(1)
                        )
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item() 

                        trange.set_postfix(
                            {'loss' : '{0:1.5f}'.format(running_loss / (step + 1))}
                        )
                        stack.append(queue.popleft())
                else:
                    break

        # oracle mode
        data = copy.deepcopy(bin_rlat_list)

        stack = []
        queue = deque(golden_edu)
        while data != [] and queue != deque([]):
            if len(stack) < 2: 
                stack.append(queue.popleft())
            else:
                if(queue != deque([])):
                    step += 1
                    self.model.zero_grad() 
                    self.optimizer.zero_grad()
                    relation = stack[len(stack)-2] + "|" + stack[len(stack)-1]
                    du = stack[len(stack)-2] + stack[len(stack)-1]

                    if relation in data: # reduce

                        sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                        sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                        sent3_torch = convert_to_bert(queue[0], tokenizer)
                        label_torch = torch.tensor([tag_to_ix['reduce']], dtype=torch.long).cuda()

                        score = self.model(
                            sent1_torch.view(1,-1),
                            sent2_torch.view(1,-1),
                            sent3_torch.view(1,-1),
                        )
                        loss = self.criterion(
                            score.view(1, len(tag_to_ix)),
                            label_torch.view(1)
                        )
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item() 

                        trange.set_postfix(
                            {'loss' : '{0:1.5f}'.format(running_loss / (step + 1))}
                        )
                        if random.uniform(0, 1) > alpha: # pick gold
                            stack.pop()
                            stack.pop()
                            stack.append(du)
                            data.remove(relation)
                        else:
                            max_score, idx = torch.max(score, 1)
                            if idx.item() == 0:
                                stack.append(queue.popleft())
                            else:
                                stack.pop()
                                stack.pop()
                                stack.append(du)
                                data.remove(relation)

                    else: # shift
                        sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                        sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                        sent3_torch = convert_to_bert(queue[0], tokenizer)
                        label_torch = torch.tensor([tag_to_ix['shift']], dtype=torch.long).cuda()

                        score = self.model(
                            sent1_torch.view(1,-1),
                            sent2_torch.view(1,-1),
                            sent3_torch.view(1,-1),
                        )
                        loss = self.criterion(
                            score.view(1, len(tag_to_ix)),
                            label_torch.view(1)
                        )
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item() 

                        trange.set_postfix(
                            {'loss' : '{0:1.5f}'.format(running_loss / (step + 1))}
                        )
                        if random.uniform(0, 1) > alpha: # pick gold
                            stack.append(queue.popleft())
                        else:
                            max_score, idx = torch.max(score, 1)
                            if idx.item() == 0:
                                stack.append(queue.popleft())
                            else:
                                stack.pop()
                                stack.pop()
                                stack.append(du)
                else:
                    break
        return running_loss, step

    def test_accuracy(self, phase, model, data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        total = n_corrects = n_wrongs = count = 0 

        trange = tqdm(enumerate(data),
                      total=len(data),
                      desc=phase,
                      ascii=True)

        self.model.eval()

        for i, (bin_r_list, gold_edu) in trange:
            data = copy.deepcopy(bin_r_list)

            stack = []
            queue = deque(gold_edu)
            while data != []:
                if len(stack) < 2: 
                    stack.append(queue.popleft())
                else:
                    if(queue != deque([])):
                        self.model.zero_grad() 
                        self.optimizer.zero_grad()
                        relation = stack[len(stack)-2] + "|" + stack[len(stack)-1]
                        du = stack[len(stack)-2] + stack[len(stack)-1]

                        if relation in data: # reduce

                            sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                            sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                            sent3_torch = convert_to_bert(queue[0], tokenizer)
                            label_torch = torch.tensor([tag_to_ix['reduce']], dtype=torch.long).cuda()

                            score = self.model(
                                sent1_torch.view(1,-1),
                                sent2_torch.view(1,-1),
                                sent3_torch.view(1,-1),
                            )
                            max_score, idx = torch.max(score, 1)

                            for j in range(0, len(idx)):
                                if idx[j] == label_torch.view(-1)[j]:
                                    n_corrects += 1
                                else:
                                    n_wrongs += 1
                            total += len(idx)

                            stack.pop()
                            stack.pop()
                            stack.append(du)
                            data.remove(relation)
                        else: # shift
                            sent1_torch = convert_to_bert(stack[len(stack)-2], tokenizer)
                            sent2_torch = convert_to_bert(stack[len(stack)-1], tokenizer)
                            sent3_torch = convert_to_bert(queue[0], tokenizer)
                            label_torch = torch.tensor([tag_to_ix['shift']], dtype=torch.long).cuda()

                            score = self.model(
                                sent1_torch.view(1,-1),
                                sent2_torch.view(1,-1),
                                sent3_torch.view(1,-1),
                            )
                            max_score, idx = torch.max(score, 1)

                            for j in range(0, len(idx)):
                                if idx[j] == label_torch.view(-1)[j]:
                                    n_corrects += 1
                                else:
                                    n_wrongs += 1
                            total += len(idx)
                            
                            stack.append(queue.popleft())
                    else:
                        break

        print("\n")
        print(total," ",n_corrects," ",n_wrongs)
        acc = float(n_corrects)/float(total)
        acc *= 100
        print("the accuracy of "+ phase + " data is: ",acc,"%")
        return acc

    def test(self):
        self.model.load_state_dict(torch.load("oracle_trans/mix_retry/model_trans.pkl.6"))
        self.model.eval()
        with torch.no_grad():
            test_acc = self.test_accuracy("test", self.model, test)

if __name__ == "__main__":
    main()


