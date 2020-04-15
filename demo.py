import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from anytree import Node,AnyNode,RenderTree,find,findall,LevelOrderIter
from anytree.exporter import DotExporter
from anytree.exporter import JsonExporter

from sklearn.metrics import precision_recall_fscore_support
from collections import deque
from tqdm import tqdm
import re
import copy
import glob
import os
import sys
import time
import math
import argparse
import json

from model_dir.model_edu_crf import NetEDU
from model_dir.model_oracle_trans import NetTrans
from model_dir.model_rlat import NetRlat

from test import buildPredictTree

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

def main(args):

    model_3 = NetEDU(768, 7, 1).cuda()
    model_1 = NetTrans(768, 2, 1).cuda()
    model_2 = NetRlat(768, 4, 4, 1).cuda()

    model_3.load_state_dict(torch.load("saved_model/EDU.pkl.19")) # load pretrained model
    model_3.eval()
    model_1.load_state_dict(torch.load("saved_model/model_trans.pkl.6")) # load pretrained model
    model_1.eval()
    model_2.load_state_dict(torch.load("saved_model/model_rlat_shallow.pkl.9")) # load pretrained model
    model_2.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    text = ''
    for line in lines:
        text += line

    predict_node_list, predict_leafnode_list = buildPredictTree(text, tokenizer, model_1, model_2, model_3, False)

    for pre, fill, node in RenderTree(predict_node_list[-1]):
        if node.is_edu == True:
            print("%s%s %s" % (pre, node.name,node.sent))
        else:
            print("%s%s" % (pre, node.name))

    # draw the picture
    # DotExporter(predict_node_list[-1]).to_picture("pred.png")

    output_data = {}
    output_data['EDUs'] = []
    output_data['trees'] = []
    output_data['relations'] = []

    for node in predict_leafnode_list:
        output_data['EDUs'].append(node.sent)

    # print(output_data)
    for node in predict_node_list:
        relation_info = {}
        arg_count = 1
        for child in node.children:
            relation_info['arg'+str(arg_count)] = child.sent
            arg_count += 1
        relation_info['sense'] = node.relation
        relation_info['center'] = node.center
        output_data['relations'].append(relation_info)

    builded_tree = []
    for node in predict_node_list:
        tree_info = {}
        tree_info['args'] = []
        for child in node.children:
            check = False
            for tree in builded_tree:
                if child.sent == tree[1]:
                    check = True
                    tree_info['args'].append(tree[0])
                    builded_tree.remove(tree)
            if check == False:
                tree_info['args'].append(child.sent)
        tree_info['sense'] = node.relation
        tree_info['center'] = node.center

        builded_tree.append((tree_info,node.sent)) # node.sent is for index

    output_data['trees'] = tree_info

    with open(args.output_file, 'w') as outfile:  
        json.dump(output_data, outfile, ensure_ascii=False)

def parse():
    parser = argparse.ArgumentParser(description="Discourse Parsing with shfit reduce method")
    parser.add_argument('--input_file', type=str, default='None', help='path of input file')
    parser.add_argument('--output_file', type=str, default='None', help='path of output file')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    main(args)