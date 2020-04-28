import copy
import os
import sys
import glob
import copy
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

from tqdm import tqdm
from preprocessor import Preprocessor
from dataset import EDUDataset, partition, EDUCollator

os.makedirs("saved_model/", exist_ok=True)

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

tagToIdx = {
    "[PAD]":0, "0":1, "1":2, "[CLS]":3, "[SEP]": 4, "[START]": 5, "[END]": 6
}

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec,1)
    return idx.item()

def logSumExp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def scalar(x):
    return x.view(-1).data.tolist()[0]

class NetEDU(nn.Module): #inherit from nn.Module
    def __init__(self, hiddenDim, tagsetSize, batchSize):
        super(NetEDU, self).__init__()
        self.hiddenDim = hiddenDim # 768
        self.batchSize = batchSize
        self.tagsetSize = tagsetSize 
        self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        # classification layer
        self.hidden2tag = nn.Linear(self.hiddenDim, self.tagsetSize) # convert to label set size
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # CRF layer
        self.transitions = nn.Parameter(torch.randn(self.tagsetSize, self.tagsetSize).cuda()) # initialize
        self.transitions.data[tagToIdx['[START]'], :] = -10000. # no transition to SOS
        self.transitions.data[:, tagToIdx['[END]']] = -10000. # no transition from EOS except to PAD
        self.transitions.data[:, tagToIdx['[PAD]']] = -10000. # no transition from PAD except to PAD
        self.transitions.data[tagToIdx['[PAD]'], :] = -10000. # no transition to PAD except from EOS
        self.transitions.data[tagToIdx['[PAD]'], tagToIdx['[END]']] = 0.
        self.transitions.data[tagToIdx['[PAD]'], tagToIdx['[PAD]']] = 0.

    def _forwardAlg(self, feats):
        mask = feats.data.gt(0).float()

        score = torch.Tensor(self.batchSize, self.tagsetSize).fill_(-10000.).cuda() # [B, C]
        score[:, 3] = 0.
        trans = self.transitions.unsqueeze(0) # [1, C, C]
        for t in range(feats.size(1)): # iterate through the sequence
            mask_t = mask[:, t]  # was mask_t = mask[:,t].unsqueeze(1), but found a bug
            emit = feats[:, t].unsqueeze(2) # [B, C, 1]
            #print(score.unsqueeze(1) + emit + trans)
            score_t = logSumExp(score.unsqueeze(1) + emit + trans) # [B, 1, C] -> [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
            # score += score_t
        score = logSumExp(score)
        return score # partition function

    def _getBertFeatures(self, input_ids1, attention_mask=None, token_type_ids=None, labels=None):
        # use sliding window approach to solve 512-constraint issue

        out_list = []
        for input_ids in [input_ids1]:
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

        sequence_output = out_list[0]
        # after combination, use a linear layer to reduce hidden dimension to tagset size
        logits = self.hidden2tag(sequence_output)
        # sequence_output, _ = self.bert(input_ids, attention_mask, output_all_encoded_layers=False)
        # sequence_output = self.dropout(sequence_output)
        # logits = self.hidden2tag(sequence_output)
        return logits

    def _scoreSentence(self, feats, tags):
 
        # initialize forward variables in log space
        mask = tags.data.gt(0).float()

        score = torch.Tensor(self.batchSize).fill_(0.).cuda()
        tags = torch.cat([torch.LongTensor(self.batchSize, 1).fill_(tagToIdx['[START]']).cuda(), tags], 1)
        #tags = torch.cat([tags,LongTensor(self.batch_size, 1).fill_(tag_to_ix[STOP_TAGP])], 1)
        feats = feats.unsqueeze(3)
        trans = self.transitions.unsqueeze(2)
        for t in range(feats.size(1)): # iterate through the sequence
            mask_t = mask[:, t]

            emit = torch.cat([feats[b, t, tags[b, t + 1]] for b in range(self.batchSize)])
            trans_t = torch.cat([trans[seq[t + 1], seq[t]] for seq in tags])
            score += (emit + trans_t) * mask_t
            # score += (emit + trans_t)

        return score

    def _viterbiDecode(self, feats):
        
        mask = feats.data.gt(0).float()

        bptr = torch.LongTensor().cuda()
        score = torch.Tensor(self.batchSize, self.tagsetSize).fill_(-10000.).cuda()
        score[:, tagToIdx['[START]']] = 0.

        for t in range(feats.size(1)): # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            bptr_t = torch.LongTensor().cuda()
            score_t = torch.Tensor().cuda()
            for i in range(self.tagsetSize): # for each next tag
                m = [j.unsqueeze(1) for j in torch.max(score + self.transitions[i], 1)]
                bptr_t = torch.cat((bptr_t, m[1]), 1) # best previous tags
                score_t = torch.cat((score_t, m[0]), 1) # best transition scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t + feats[:, t] # plus emission scores
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batchSize):
            x = best_tag[b] # best tag
            l = int(scalar(mask[b].sum()))
            for bptr_t in reversed(bptr[b][:l]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()
        #print(best_path)
        return best_path

    def negLogLikelihood(self, X, tags, X_mask=None):
        feats = self._getBertFeatures(X, X_mask)
        forwardScore = self._forwardAlg(feats)
        goldScore = self._scoreSentence(feats, tags)

        return forwardScore - goldScore

    def forward(self, X, X_mask=None):
        # Get the emission scores from the BiLSTM
        bertFeats = self._getBertFeatures(X, X_mask)

        # Find the best path, given the features.
        #score, tag_seq = self._viterbi_decode(lstm_feats)
        #return score, tag_seq
        path = self._viterbiDecode(bertFeats)  
        return bertFeats, path


class ModelEDU():
    def __init__(self, trainData, testData, validData, 
                 embeddingDim, tagsetSize, 
                 batchSize):

        self.trainData = trainData
        self.testData = testData
        self.validData = validData
        self.embeddingDim = embeddingDim # 768(BERT)
        self.tagsetSize = tagsetSize
        self.batchSize = batchSize

        self.model = NetEDU(self.embeddingDim, self.tagsetSize, self.batchSize).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr= 5e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def train(self):

        collateFn = EDUCollator()

        train = DataLoader(
            self.trainData,
            batch_size=self.batchSize,
            collate_fn=collateFn,
        )
        valid = DataLoader(
            self.validData,
            batch_size=self.batchSize,
            collate_fn=collateFn
        )
        test = DataLoader(
            self.testData,
            batch_size=self.batchSize,
            collate_fn=collateFn
        )

        for epoch in range(40): 
            runningLoss = 0.0
            self.model.train()

            trange = tqdm(enumerate(train),
                          total=len(train),
                          desc='ModelEDU Training',
                          ascii = True)

            for step, (data, mask, label) in trange:
                # if step == 10:
                #     break
                dataTorch = torch.tensor(data, dtype=torch.long).cuda()
                maskTorch = torch.tensor(mask, dtype=torch.long).cuda()
                labelTorch = torch.tensor(label, dtype=torch.long).cuda()

                self.model.zero_grad()

                out = self.model(dataTorch, maskTorch)
                loss = self.model.negLogLikelihood(dataTorch, labelTorch, maskTorch)
                loss = torch.mean(loss)
                loss.backward()
                self.optimizer.step()

                runningLoss += loss.item()

                trange.set_postfix(
                    {'loss' : '{0:1.5f}'.format(runningLoss / (step + 1))}
                )

            print("\n")
            print('[%d] loss: %.5f' %
                  (epoch + 1, runningLoss / len(train)))

            # for early stopping
            with torch.no_grad():
                self.testAccuracy("train", self.model, train)

            with torch.no_grad():
                self.testAccuracy("dev", self.model, valid)

            torch.save(self.model.state_dict(),'saved_model/model_edu.pkl.{}'.format(epoch+1))

    def testAccuracy(self, phase, model, inputData):
        total = nCorrects = nWrongs = count = 0 

        trange = tqdm(enumerate(inputData),
                      total=len(inputData),
                      desc=phase,
                      ascii = True)

        self.model.eval()
        for step, (data, mask, label) in trange:
            # if step == 10:
            #     assert 1 == 0
            dataTorch = torch.tensor(data, dtype=torch.long).cuda()
            maskTorch = torch.tensor(mask, dtype=torch.long).cuda()
            labelTorch = torch.tensor(label, dtype=torch.long).cuda()
            
            logits, path = self.model(dataTorch, maskTorch)

            for j in range(1, len(path[0])-1):
                if(path[0][j] != 1 and path[0][j] != 2):
                    maxScore, idx = torch.max(logits[:,j,1:3], -1)
                    path[0][j] = idx.item()+1
                if path[0][j] == labelTorch.view(-1)[j]:
                    nCorrects += 1
                else:
                    nWrongs += 1

            total += len(path[0])-2

        print("\n")
        print(total, " ", nCorrects," ", nWrongs)
        acc = float(nCorrects) / float(total)
        acc *= 100
        print("The accuracy of " + phase + " data is: ", acc, "%")


    def test(self):

        self.model.load_state_dict(torch.load("saved_model/pretrained_edu.pkl")) # load pretrained model
        self.model.eval()

        with torch.no_grad():
            self.testAccuracy("train", self.model, train)
        with torch.no_grad():
            self.testAccuracy("dev", self.model, valid)
        with torch.no_grad():
            self.testAccuracy("test", self.model, test)


if __name__ == "__main__":
    main()


