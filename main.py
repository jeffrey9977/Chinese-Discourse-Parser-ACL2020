import argparse
import os 
import glob
import pandas as pd
import numpy as np
from preprocessor import Preprocessor

def main(args, k_fold=23, batchSize=1):

    if args.make_dataset:
        # training data : RST-style CDTB
        # augmentation data for UDA : PDTB-style CDTB
        preprocessor = Preprocessor()

        testEdu, testTrans, testRlat = \
            preprocessor.makeDatasetCdtb(
                '../CDTB/test', 'csv', makeDev=False, oversample=False
            ) 
        # testRlatAug = \
        #     preprocessor.makeDatasetUda(
        #         '../UDA/test', 'json', oversample=False
        #     ) 
        trainRlatAug = \
            preprocessor.makeDatasetUda(
                '../UDA/train', 'json', oversample=False
            )
        trainEdu, trainTrans, trainRlat, validEdu, validTrans, validRlat = \
            preprocessor.makeDatasetCdtb(
                '../CDTB/train', 'csv', makeDev=True, oversample=False
            )

    if args.train_edu:
        from model_dir.model_edu_crf import ModelEDU
        model = ModelEDU(
                    trainEdu, testEdu, validEdu, embeddingDim=768, 
                    tagsetSize=7, batchSize=batchSize
                )
        model.train()

    if args.train_trans:
        from model_dir.model_oracle_trans import ModelTrans
        model = ModelTrans(
                    trainTrans, testTrans, validTrans, embedding_dim=768, 
                    tagset_size=2, batch_size=batchSize
                )
        model.train()

    if args.train_rlat:
        from model_dir.model_rlat import ModelRlat
        model = ModelRlat(
                    trainRlat, testRlat, trainRlatAug, validRlat,
                    embeddingDim=768, tagsetSizeCenter=3, tagsetSizeRelation=4, 
                    tagsetSizeSubLabel=17, batchSize=batchSize
                )
        model.train()

def parse():
    parser = argparse.ArgumentParser(description="CDTB discourse parsing with shfit reduce method and use PDTB-style CDTB as augmentation data")
    parser.add_argument('--make_dataset',action='store_true',help='whether make training dataset')
    parser.add_argument('--train_edu', action='store_true', help='whether train edu segmenter')
    parser.add_argument('--train_trans', action='store_true', help='whether train shift & reduce parser')
    parser.add_argument('--train_rlat', action='store_true', help='whether train relation and center classifier')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    main(args)

