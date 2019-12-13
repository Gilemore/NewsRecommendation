# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import evaluation
import baselines
import argparse

parser = argparse.ArgumentParser(description='Data path info!')
#parser.add_argument("--a", default=1, type=int, help="This is the 'a' variable")
parser.add_argument("--Path", required=True, type=str, help="Main path")
parser.add_argument("--Train", required=True, type=str, help="PATH to train_data")
parser.add_argument("--Test", required=True, type=str, help="PATH to test_data")
parser.add_argument("--Prediction", required=True, type=str, help="category or article")
args = parser.parse_args()

Path=args.Path
PATH_TO_TRAIN = args.Train
PATH_TO_TEST = args.Test
Prediction=args.Prediction
import sys
sys.path.append(Path+"/baseline")
import evaluation
import baselines
import sknn_1_features
import vmsknn
import iknn


if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})

    #RP=baselines.RandomPred()
    #RP.fit(data)
    #res = evaluation.evaluate_sessions(RP,valid,data)
    #print("The accuracy of Random Prediction:")
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
    #print("\n")

    #BRP=baselines.BPR(data)
    #BRP.fit(data)
    #res = evaluation.evaluate_sessions(BRP,valid,data)
    #with open(Path+'/wrong/BRP_wrong.txt', 'w') as filehandle:
    #  filehandle.writelines("{}\n".format(i) for i in res[2])
    #print("The accuracy of Bayesian Personalized Ranking Matrix Factorization:")
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
    #print("\n")

    iknn=vmsknn.VMSessionKNN(data)
    iknn.fit(data)
    res = evaluation.evaluate_sessions(iknn,valid,data)
    print("The accuracy of item k-nearest neighbors:")
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print("\n")


    #sknn=sknn_1_features.SessionKNN(data,other_variables_1=False)
    #sknn.fit(data)
    #res = evaluation.evaluate_sessions(sknn,valid,data)
    #print("The accuracy of session k-nearest neighbors:")
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1])) 
    #print("\n")

    #sknn=sknn_1_features.SessionKNN(data,other_variables_1=True,other_variables_2=False)
    #sknn.fit(data)
    #res = evaluation.evaluate_sessions(sknn,valid,data)
    #print("The accuracy of session k-nearest neighbors with feature region:")
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1])) 
    #print("\n")

    
    #sknn=sknn_1_features.SessionKNN(data)
    #sknn.fit(data)
    #res = evaluation.evaluate_sessions(sknn,valid,data)
    #print("The accuracy of session k-nearest neighbors with features region and devive type:")
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
    

    
