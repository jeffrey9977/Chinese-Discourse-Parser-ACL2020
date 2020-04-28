import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
import numpy as np
import math
import copy
import csv
import os

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from dataset import partition, RlatCollator, RlatDataset, AugCollator, AugDataset

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

os.makedirs("saved_model/", exist_ok=True)

tag_to_ix_relation = {'causality':0, 'coordination':1, 'transition':2, 'explanation':3}

tag_to_ix_center = {'1':0, '2':1, '3':2, '4':3}
# tag_to_ix_center = {'1':0, '2':1, '3':2}

class NetRlat(nn.Module):

    def __init__(self, embedding_dim, tagset_size_center,tagset_size_relation,batch_size):
        super(NetRlat,self).__init__()

        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size_center = tagset_size_center # center label size
        self.tagset_size_relation = tagset_size_relation # relation label size
        self.batch_size = batch_size
        # BERT
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').cuda()
        self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        # freeze
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.hidden2tag_center = nn.Linear(embedding_dim*2, self.tagset_size_center)
        self.hidden2tag_relation = nn.Linear(embedding_dim*2, self.tagset_size_relation)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,input_ids1,input_ids2,aug_flag=None,input_mask1=None,input_mask2=None,labels=None):
        '''
            Args:
                input_ids: [batch_size, seq_length]
        '''

        # use sliding window approach to deal with BERT length 512 restriction
        # use narrow(dimension, start, length) to slice tensor 
        out_list = []
        for input_ids in [input_ids1, input_ids2]:
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
        center = self.hidden2tag_center(pooled)
        relation = self.hidden2tag_relation(pooled)
        if aug_flag == True:
            center = self.softmax(center)
            relation = self.softmax(relation)
        elif aug_flag == False:
            center = self.logsoftmax(center)
            relation = self.logsoftmax(relation)
        else:
            center = center
            relation = relation
            
        return center, relation

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.reduce = reduce

    def forward(self, inputs, targets):
        CE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class ModelRlat():
    def __init__(self, train_data, test_data, aug_data, valid_data, embedding_dim, tagset_size_center, tagset_size_relation, tagset_size_sub_label, batch_size, k_fold):

        self.train_data = train_data
        self.test_data = test_data
        self.aug_data = aug_data
        self.valid_data = valid_data
        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size_center = tagset_size_center # center label size
        self.tagset_size_relation = tagset_size_relation # relation label size
        self.batch_size = batch_size
        self.k_fold = k_fold   

        self.model = _ModelRlat(self.embedding_dim,self.tagset_size_center,self.tagset_size_relation, self.batch_size).cuda()
        self.optimizer = optim.SGD(self.model.parameters(),lr= 5e-5)
        # self.optimizer = BertAdam(self.model.parameters(),lr= 1e-4)
        self.ce_criterion = nn.CrossEntropyLoss().cuda() # delete ignore_index = 0 
        # self.ce_criterion = FocalLoss().cuda()
        self.kl_criterion = nn.KLDivLoss().cuda()
        # self.kl_criterion = nn.MSELoss().cuda()

    def train(self):
        # indices = list(range(len(self.train_data)))
        # np.random.shuffle(indices)

        # partitions = list(partition(indices, self.k_fold))

        # train_idx = [idx for part in partitions[0:self.k_fold-1] for idx in part]
        # valid_idx = partitions[self.k_fold-1]

        # # randomly sample from only the indicies given
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        collate_fn_rlat = RlatCollator(train_edu=False, train_trans=False, train_rlat=True)
        collate_fn_aug = AugCollator(train_trans=False, train_rlat=True)

        train_data = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            # sampler=train_sampler,
            collate_fn=collate_fn_rlat
        )
        valid_data = DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            # sampler=valid_sampler,
            collate_fn=collate_fn_rlat
        )
        test_data = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn_rlat
        )
        aug_data = DataLoader(
            self.aug_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn_aug
        )

        for epoch in range(15): 
            running_loss1 = 0.0
            running_loss2 = 0.0  
            running_loss3 = 0.0            
            running_loss4 = 0.0            

            self.model.train()
            i = 0

            trange = tqdm(enumerate(zip(train_data, aug_data)),
                          total=len(aug_data),
                          desc='rlat')

            for step, (train, aug) in trange:

                self.model.zero_grad() 
                self.optimizer.zero_grad()
                # if step == 10:
                #     break
                zh1, zh2, relation, center, sub_label = train[0], train[1], train[2], train[3], train[4]
                en1, en2 = aug[0], aug[1]
                aug_en1, aug_en2 = aug[4], aug[5]

                zh1_torch = torch.tensor(zh1, dtype=torch.long).cuda()
                zh2_torch = torch.tensor(zh2, dtype=torch.long).cuda()

                relation_torch = torch.tensor([relation], dtype=torch.long).cuda()
                center_torch = torch.tensor([center], dtype=torch.long).cuda()

                en1_torch = torch.tensor(en1, dtype=torch.long).cuda()
                en2_torch = torch.tensor(en2, dtype=torch.long).cuda()

                aug_en1_torch = torch.tensor(aug_en1, dtype=torch.long).cuda()
                aug_en2_torch = torch.tensor(aug_en2, dtype=torch.long).cuda()

                center_zh, relation_zh = self.model(
                    zh1_torch.view(self.batch_size,-1),
                    zh2_torch.view(self.batch_size,-1),
                )
                center_en, relation_en = self.model(
                    en1_torch.view(self.batch_size,-1),
                    en2_torch.view(self.batch_size,-1),
                    aug_flag=False,
                )
                center_aug_en, relation_aug_en = self.model(
                    aug_en1_torch.view(self.batch_size,-1),
                    aug_en2_torch.view(self.batch_size,-1),
                    aug_flag=True,
                )
                # supervised cross-entropy loss
                ce_relation_loss = self.ce_criterion(
                    relation_zh.view(self.batch_size, self.model.tagset_size_relation),
                    relation_torch.view(self.batch_size)
                )
                ce_center_loss = self.ce_criterion(
                    center_zh.view(self.batch_size,self.model.tagset_size_center),
                    center_torch.view(self.batch_size)
                )

                # unsupervised consistency loss (kl-divergence)
                kl_relation_loss = self.kl_criterion(
                    relation_en.view(self.batch_size, self.model.tagset_size_relation),
                    relation_aug_en.view(self.batch_size, self.model.tagset_size_relation)
                )
                kl_center_loss = self.kl_criterion(
                    center_en.view(self.batch_size, self.model.tagset_size_center),
                    center_aug_en.view(self.batch_size, self.model.tagset_size_center)
                )
                # Training Signal Annealing
                center_loss = 0.0
                relation_loss = 0.0

                # center_thresold = (1 - math.exp(-( (step+1) / len(train_data))*5)) * (1 - 1 / self.model.tagset_size_center) + (1 / self.model.tagset_size_center)
                # relation_thresold = (1 - math.exp(-( (step+1) / len(train_data))*5)) * (1 - 1 / self.model.tagset_size_relation) + (1 / self.model.tagset_size_relation)

                # if center_zh[:, center].item() > center_thresold:
                #     center_loss = kl_center_loss
                # else:
                #     center_loss = ce_center_loss + kl_center_loss

                # if relation_zh[:, relation].item() > relation_thresold:
                #     relation_loss = kl_relation_loss
                # else:
                #     relation_loss = ce_relation_loss + kl_relation_loss

                center_loss = ce_center_loss + 10 * kl_center_loss
                relation_loss = ce_relation_loss + 10 * kl_relation_loss

                loss = []
                loss.append(center_loss)
                loss.append(relation_loss)

                gradients = [torch.tensor(1.0).cuda() for _ in range(len(loss))]
                torch.autograd.backward(loss, gradients)            
                # loss = center_loss + relation_loss
                # loss.backward()
                self.optimizer.step()

                running_loss1 += ce_center_loss.item()
                running_loss2 += kl_center_loss.item()
                running_loss3 += ce_relation_loss.item()
                running_loss4 += kl_relation_loss.item()

                trange.set_postfix(
                    {'ce_c' : '{0:1.5f}'.format(running_loss1 / (step + 1)),
                     'kl_c' : '{0:1.5f}'.format(running_loss2 / (step + 1)),
                     'ce_s' : '{0:1.5f}'.format(running_loss3 / (step + 1)),
                     'kl_s' : '{0:1.5f}'.format(running_loss4 / (step + 1))
                    }
                )

            print("\n")
            print('[%d] loss of ce_center: %.5f' %
                  (epoch + 1, running_loss1 * self.batch_size / len(train_data)))
            print('[%d] loss of kl_center: %.5f' %
                  (epoch + 1, running_loss2 * self.batch_size / len(train_data)))
            print('[%d] loss of ce_relation: %.5f' %
                  (epoch + 1, running_loss3 * self.batch_size / len(train_data)))
            print('[%d] loss of kl_relation: %.5f' %
                  (epoch + 1, running_loss4 * self.batch_size / len(train_data)))


                # running_loss1 += loss[0].item()
                # running_loss2 += loss[1].item()

                # trange.set_postfix(
                #     {'center_loss' : '{0:1.5f}'.format(running_loss1 / (step + 1)),
                #      'relation_loss' : '{0:1.5f}'.format(running_loss2 / (step + 1))
                #     }
                # )

            # print("\n")
            # print('[%d] loss of center: %.5f' %
            #       (epoch + 1, running_loss1 * self.batch_size / len(train_data)))
            # print('[%d] loss of relation: %.5f' %
            #       (epoch + 1, running_loss2 * self.batch_size / len(train_data)))

            with torch.no_grad():
                train_acc = self.test_accuracy("train", train_data)
            with torch.no_grad():
                valid_acc =  self.test_accuracy("valid", valid_data)

            torch.save(self.model.state_dict(),'saved_model/model_rlat.pkl.{}'.format(epoch+1))

        with torch.no_grad():
            test_acc = self.test_accuracy("test", test_data)

    def test(self):
        self.model.load_state_dict(torch.load("saved_model/pretrained_rlat.pkl")) # load pretrained model
        self.model.eval()

        collate_fn_rlat = RlatCollator(train_edu=False, train_trans=False, train_rlat=True)
        test_data = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn_rlat
        )
        # with torch.no_grad():
        #     self.test_accuracy("train", train_data)
        # with torch.no_grad():
        #     self.test_accuracy("valid", valid_data)
        with torch.no_grad():
           test_acc = self.test_accuracy("test", test_data)

    def test_accuracy(self,phase,data):
        l0 = l1 = l2 = l3 = 0
        t0 = t1 = t2 = t3 = 0

        c_l0 = c_l1 = c_l2 = c_l3 = 0
        c_t0 = c_t1 = c_t2 = c_t3 = 0

        total = n_corrects = n_wrongs = count = 0 

        causality2explanation = explanation2causality = 0

        trange = tqdm(enumerate(data),
                      total=len(data),
                      desc=phase)

        self.model.eval()
        for step, (sent1, sent2, relation, center, _) in trange:
            # if step == 10:
            #     break
            sent1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
            sent2_torch = torch.tensor(sent2,dtype=torch.long).cuda()

            # mask1_torch = torch.tensor(mask1,dtype=torch.long).cuda()
            # mask2_torch = torch.tensor(mask2,dtype=torch.long).cuda()

            relation_torch = torch.tensor([relation], dtype=torch.long).cuda()
            center_torch = torch.tensor([center], dtype=torch.long).cuda()

            center, relation = self.model(sent1_torch.view(self.batch_size,-1),sent2_torch.view(self.batch_size,-1))
            
            max_score_relation, relation_idx = torch.max(relation, 1)
            max_score_center, center_idx = torch.max(center, 1)

            for j in range(0, len(relation_idx)):
                if relation_idx[j] == relation_torch.view(-1)[j] and center_idx[j] == center_torch.view(-1)[j]:
                    n_corrects += 1
                else:
                    n_wrongs += 1

            total += len(relation_idx)

            for j in range(0, len(relation_idx)):
                if relation_idx[j] == 0:
                    t0 +=1
                if relation_idx[j] == 1:
                    t1 +=1
                if relation_idx[j] == 2:
                    t2 +=1
                if relation_idx[j] == 3:
                    t3 +=1

            for j in range(0, len(relation_idx)):
                if relation_torch.view(-1)[j] == 0:
                    l0 +=1
                if relation_torch.view(-1)[j] == 1:
                    l1 +=1
                if relation_torch.view(-1)[j] == 2:
                    l2 +=1
                if relation_torch.view(-1)[j] == 3:
                    l3 +=1

            for j in range(0, len(center_idx)):
                if center_idx[j] == 0:
                    c_t0 +=1
                if center_idx[j] == 1:
                    c_t1 +=1
                if center_idx[j] == 2:
                    c_t2 +=1
                if center_idx[j] == 3:
                    c_t3 +=1

            for j in range(0, len(center_idx)):
                if center_torch.view(-1)[j] == 0:
                    c_l0 +=1
                if center_torch.view(-1)[j] == 1:
                    c_l1 +=1
                if center_torch.view(-1)[j] == 2:
                    c_l2 +=1
                if center_torch.view(-1)[j] == 3:
                    c_l3 +=1

            for j in range(0, len(relation_idx)):
                if relation_idx[j] ==  0 and relation_torch.view(-1)[j] == 3:
                    causality2explanation +=1 
                if relation_idx[j] ==  3 and relation_torch.view(-1)[j] == 1:
                    explanation2causality += 1

        print("\n")

        print('causality    = ',t0," ans = ",l0)
        print('coordination = ',t1," ans = ",l1)
        print('transition   = ',t2," ans = ",l2)
        print('explanation  = ',t3," ans = ",l3)

        print('front = ',c_t0," ans = ",c_l0)
        print('back  = ',c_t1," ans = ",c_l1)
        print('equal = ',c_t2," ans = ",c_l2)
        print('multi = ',c_t3," ans = ",c_l3)

        print('causality2explanation = ', causality2explanation)
        print('explanation2causality = ', explanation2causality)

        print("\n")
        print(total," ",n_corrects," ",n_wrongs)
        acc = float(n_corrects)/float(total)
        acc *= 100
        print("the accuracy of "+ phase + " data is: ",acc,"%")
        return acc
