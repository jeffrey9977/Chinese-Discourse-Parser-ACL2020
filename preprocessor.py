import pandas
import numpy
import json
import glob
import copy
import random
import math
import sys
from tqdm import tqdm
from collections import deque
from torch.utils.data import Dataset
from dataset import EDUDataset, TransDataset, RlatDataset, AugDataset
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

class Preprocessor():

    def __init__(self):

        self.eduDict = { 
            "[PAD]":0, "0":1, "1":2, "[CLS]":3, "[SEP]": 4 
        }
        self.eduCrfDict = {
            "[PAD]":0, "0":1, "1":2, "[CLS]":3, "[SEP]": 4, "[START]": 5, "[END]": 6
        }
        self.transDict = { 
            "shift":0, "reduce":1 
        }
        self.rlatDictCdtb = {
            'causality':0, 'coordination':1, 'transition':2, 'explanation':3
        } 
        self.subRlatDictCdtb = {
            '因果关系':0, '背景关系':1, '目的关系':2, '条件关系':3, '假设关系':4, '推断关系':5, 
            '例证关系':6, '解说关系':7, '总分关系':8, '评价关系':9,
            '并列关系':10, '顺承关系':11, '对比关系':12, '递进关系':13, '选择关系':14,
            '让步关系':15, '转折关系':16
        } # causality, explanation, coordination, transition

        # self.centerDict = {
        #     '1':0, '2':1, '3':2 
        # }
        self.langDict = {
            'ch':0, 'en':1
        }
        self.centerDict = {
            '1':0, '2':1, '3':2, '4':3 
        }
        self.traindataEdu = []
        self.traindataTrans = []
        self.traindataRlat = [] 

    def makeDatasetCdtb(self, fileDir, extension, makeDev=False, oversample=False):
        # calculate the size of augmented sentence pairs
        # then we need to let our training data match this size
        if fileDir == 'train':
            rlatPdtbLen = len(self.traindataRlat)

        self.traindataDdu = []
        self.traindataTrans = []
        self.traindataRlat = []
        self.numData = 0
        self.traindataEduCount = 0

        allFiles = [f for f in sorted(glob.glob('{}/*.{}'.format(fileDir, extension)))]
        
        trange = tqdm(enumerate(allFiles),
                  total=len(allFiles),
                  desc=fileDir,
                  ascii=True)
        # iterate through all training data(csv)
        for i, file in trange:
            trange.set_description("Processing %s" % file)
            df = dataframe = pandas.read_csv(file)
            # add label (relation type + center)
            df['label'] = df['RelationType'].map(str) + '_' + df['Center'].map(str)

            for idx in set(df['p_id'].tolist()): 
                # get a paragraph everytime
                paragraph = df[df['p_id']==idx]
                # format of s_list : [ 's1|s2', 's3|s4',...... ]
                paragraph = paragraph['Sentence'].tolist()

                if paragraph == []: 
                    continue
                # generate golden EDU
                goldenEdu = self.genGoldenEdu(paragraph)
                # generate EDU training data
                self.genTraindataEdu(goldenEdu)
                # generate trans training data
                # self.genTraindataTrans(goldenEdu, paragraph, self.traindataTrans)
                self.genTraindataOracleTrans(goldenEdu, paragraph, self.traindataTrans)
            # generate rlat training data
            self.genTraindataRlatCdtb(df, self.traindataRlat, oversample)

        # total = count1 = count2 = count3 = count4 = 0
        # for sent1, sent2, label in self.traindata_rlat:
        #     if label.split('_')[1] == '1':
        #         count1 += 1
        #     if label.split('_')[1] == '2':
        #         count2 += 1
        #     if label.split('_')[1] == '3':
        #         count3 += 1
        #     if label.split('_')[1] == '4':
        #         count4 += 1
        #     total += 1
        # print('count1: ', count1)
        # print('count2: ', count2)
        # print('count3: ', count3)
        # print('count4: ', count4)
        # print('total : ', total)

        if makeDev:
            dataEdu = copy.deepcopy(self.traindataEdu)
            random.shuffle(dataEdu)
            cut = math.ceil(len(dataEdu)*0.9) # 90% train, 10% valid
            traindataEdu = dataEdu[:cut]
            validdataEdu = dataEdu[cut:]
            # trans
            dataTrans = copy.deepcopy(self.traindataTrans)
            random.shuffle(dataTrans)
            cut = math.ceil(len(dataTrans)*0.9) # 90% train, 10% valid
            traindataTrans = dataTrans[:cut]
            validdataTrans = dataTrans[cut:]
            # rlat
            dataRlat = copy.deepcopy(self.traindataRlat)
            random.shuffle(dataRlat)
            cut = math.ceil(len(dataRlat)*0.9) # 90%
            traindataRlat = dataRlat[:cut]
            validdataRlat = dataRlat[cut:]

            # to match aug_data size
            if fileDir == 'train':
                idx = 0
                for i in range(rlatPdtbLen-len(traindataRlat)):
                    traindataRlat.append(traindataRlat[idx])
                    idx += 1
                assert traindataRlat[0] == traindataRlat[8083] # no oversample == 8083 0.9 7185 0.8

            print('len of {}dataEdu    : {} '.format(fileDir, len(traindataEdu)))
            print('len of {}dataTrans  : {} '.format(fileDir, len(traindataTrans)))
            print('len of {}dataRlat   : {} '.format(fileDir, len(traindataRlat)))
            print('len of {}dataEdu    : {} '.format('valid', len(validdataEdu)))
            print('len of {}dataTrans  : {} '.format('valid', len(validdataTrans)))
            print('len of {}dataRlat   : {} '.format('valid', len(validdataRlat)))           

            return EDUDataset(traindataEdu, self.eduCrfDict), \
                   TransDataset(traindataTrans), \
                   RlatDataset(traindataRlat, self.rlatDictCdtb, self.centerDict, self.subRlatDictCdtb), \
                   EDUDataset(validdataEdu, self.eduCrfDict), \
                   TransDataset(validdataTrans), \
                   RlatDataset(validdataRlat, self.rlatDictCdtb, self.centerDict, self.subRlatDictCdtb)
        else:
            traindataEdu = copy.deepcopy(self.traindataEdu)
            traindataTrans = copy.deepcopy(self.traindataTrans)
            traindataRlat = copy.deepcopy(self.traindataRlat)

            print('len of {}dataEdu    : {} '.format(fileDir, len(self.traindataEdu)))
            print('len of {}dataTrans  : {} '.format(fileDir, len(self.traindataTrans)))
            print('len of {}dataRlat   : {} '.format(fileDir, len(self.traindataRlat)))

            return EDUDataset(traindataEdu, self.eduCrfDict), \
                   TransDataset(traindataTrans), \
                   RlatDataset(traindataRlat, self.rlatDictCdtb, self.centerDict, self.subRlatDictCdtb)

    def makeDatasetUda(self, fileDir, extension, oversample=False):
        """
            this is for Unsupervised Data Augmentation
        """
        self.traindataEdu = []
        self.traindataTrans = []
        self.traindataRlat = [] 
        self.numData = 0

        # shallow relation (PDTB-style CDTB from conll2016)
        allFiles = [f for f in sorted(glob.glob('{}/*.{}'.format(fileDir, 'json')))]
        # iterate through all training data(csv)
        for file in allFiles:
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line) 
                    # generate rlat training data
                    self.genTraindataRlatShallow(data, self.traindataRlat, oversample, file)

        print('len of {}dataRlat  : {} '.format(fileDir, len(self.traindataRlat)))

        traindataRlat = copy.deepcopy(self.traindataRlat)

        return AugDataset(traindataRlat)

    def genGoldenEdu(self, paragraph):
        """ 
            segment the paragraph with golden standard '|'
            and store those sentences(EDU) in list p
            available both on CDTB and RSTDT

        """

        data = paragraph.copy()
        p = data[0].split("|")
        data.remove(data[0])

        while data != []:
            for sent in data:
                for sent2 in p:
                    if sent.replace("|","") == sent2:
                        data.remove(sent)
                        for idx,item in enumerate(sent.split("|")):
                            p.insert(p.index(sent2)+1+idx,item)
                        p.remove(sent2)
        return p


    def genTraindataEdu(self, goldEDUs):
        '''
            generate training data by labeling every word with 0(I) or 1(B) 
            1 if this word is the begining of the EDU
            0 otherwise
        '''

        p = copy.deepcopy(goldEDUs)
        # use basic tokenizer instead of bert tokenizer because bert is character
        # but we are labeling word
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.traindataEdu.append([])  
        self.traindataEdu[self.traindataEduCount].append(['[CLS]'])
        self.traindataEdu[self.traindataEduCount].append(['[CLS]'])

        for sent in p:
            for j, word in enumerate(list(sent)): # split each word with space
                self.traindataEdu[self.traindataEduCount][0].append(word)
                if j == 0:
                    self.traindataEdu[self.traindataEduCount][1].append("1")
                else:
                    self.traindataEdu[self.traindataEduCount][1].append("0")
        self.traindataEdu[self.traindataEduCount][0].append('[SEP]')
        self.traindataEdu[self.traindataEduCount][1].append('[SEP]')

        self.traindataEduCount += 1
        return

    def genTraindataOracleTrans(self, goldEdu, rList, traindataTrans):
        """
            generate transition actions training data "per paragraph"
            because we are going to train with dynamic oracle technique 
            [v1 from stack, v2 from stack, v3 from queue ] -> label(shift or reduce)

            Args:
                goldEdu : document segmented as EDUs
                rList : ['s1|s2', 's2|s3|s4']
                traindataTrans : preprocessed training data of trans

        """

        # create binary relation list
        binRList = self.genBinaryAction(goldEdu, rList)
        traindataTrans.append([binRList, goldEdu])
        return 

    def genTraindataRlatCdtb(self, df, traindataRlat, oversample):

        info = df[['p_id','r_id','RelationType','Sentence','Center','label']]

        for idx in range(len(info)):

            sent = info.iloc[idx,3].split("|")
            # coordination 
            if len(sent) >= 3:
                for idx1 in range(len(sent)-1):
                    for idx2 in range(idx1+1,len(sent)):
                        newRelation = self.changeRelationCdtb(info.iloc[idx,2])
                        traindataRlat.append([sent[idx1],sent[idx2],newRelation + "_" + str(info.iloc[idx,4]),info.iloc[idx,2]])
                        self.numData += 1
            else:
                newRelation = self.changeRelationCdtb(info.iloc[idx,2])
                traindataRlat.append([sent[0],sent[1],newRelation + "_" + str(info.iloc[idx,4]),info.iloc[idx,2]])
                self.numData += 1 
                # over sample 
                if traindataRlat[self.numData-1][2].split("_")[0] != "coordination" and oversample:

                    dataLabel = traindataRlat[self.numData-1][2].split("_")
                    newLabel = ""
                    if dataLabel[1] == "1":
                        newLabel = dataLabel[0] + "_" + "2"
                    elif dataLabel[1] == "2":
                        newLabel = dataLabel[0] + "_" + "1"
                    else:
                        newLabel = dataLabel[0] + "_" + "3"

                    traindataRlat.append([traindataRlat[self.numData-1][1], traindataRlat[self.numData-1][0], newLabel])
                    self.numData += 1         

    def genTraindataRlatShallow(self, jsonLine, traindataRlat, oversample, file):
        """
            generate relation type training data
            [v1 from stack, v2 from stack] -> label(relation type and center)

            Args:
                json_line : document segmented as EDUs
                train_data_ : preprocessed training data of trans
                oversample
        """

        sent = jsonLine['Sentence'].split("|")
        augSent = jsonLine['aug_Sentence'].split("|")
        # we wont use the relation label of augmentation data 
        # so temporarily set as 'Explanation_3'
        if not (len(sent[0]) > 510 or len(sent[1]) > 510 or len(augSent[0]) > 510 or len(augSent[1]) > 510):
            traindataRlat.append([sent[0],sent[1],'Explanation_3', augSent[0],augSent[1]])
        self.numData += 1 

        # over sample 
        # if train_data_rlat[self.num_data-1][2].split("_")[0] != "coordination" and oversample:

        #     data_label = train_data_rlat[self.num_data-1][2].split("_")
        #     new_label = ""
        #     if data_label[1] == "1":
        #         new_label = data_label[0] + "_" + "2"
        #     elif data_label[1] == "2":
        #         new_label = data_label[0] + "_" + "1"
        #     else:
        #         new_label = data_label[0] + "_" + "3"

        #     trainDataRlat.append([trainDataRlat[self.num_data-1][1],trainDataRlat[self.num_data-1][0],new_label])
        #     self.num_data += 1 

    def genBinaryAction(self, p, sList):
        # ex: 1|2|3   --> 1|2 , 12|3
        #     1|2|3|4 --> 1|2 , 12|3 , 123|4
        data = sList.copy()

        for sent in data:
            if len(sent.split("|")) >= 3:
                nodes = sent.split("|")
                for idx in range(0,len(nodes)-1):
                    data.insert(data.index(sent)+1+idx,nodes[0]+"|"+nodes[1])
                    new_node = nodes[0]+nodes[1]
                    nodes.remove(nodes[1])
                    # if idx == 0:
                    nodes.remove(nodes[0])
                    nodes.insert(0,new_node)

                data.remove(sent)

        binSList = data.copy()
        return binSList

    def changeRelationCdtb(self,relationType):

        if relationType in ['因果关系','背景关系','目的关系','条件关系','假设关系','推断关系']:
            relationType = 'causality'
        elif relationType in ['例证关系','解说关系','总分关系','评价关系']:
            relationType = 'explanation'
        elif relationType in ['并列关系','顺承关系','对比关系','递进关系','选择关系']:
            relationType = 'coordination'
        elif relationType in ['让步关系','转折关系']:
            relationType = 'transition'

        return relationType