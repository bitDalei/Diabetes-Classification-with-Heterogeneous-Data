import sklearn
from utils.utils import *
import pandas as pd
import numpy as np

from nets import CNN_CNN,CNN_Inception,CNN_MLP
from nets import LSTM_CNN,LSTM_Inception,LSTM_MLP
from nets import Resnet_CNN,Resnet_Inception,Resnet_MLP

import warnings

def evaluate_model(x_train_cgm, x_train_attr, x_test_cgm, x_test_attr, y_train, y_test, approach, flag_cbam, flag_att, output_dir, data_flag='both'):

    num_classes = len(np.unique(np.concatenate((y_train,y_test), axis=0)))

    y_true = y_test

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train_cgm.shape) == 2:
        print('x_train is univariate')
        x_train_cgm = x_train_cgm.reshape((x_train_cgm.shape[0], x_train_cgm.shape[1], 1))
        x_test_cgm = x_test_cgm.reshape((x_test_cgm.shape[0], x_test_cgm.shape[1], 1))
        x_train_attr = x_train_attr.reshape((x_train_attr.shape[0], x_train_attr.shape[1], 1))
        x_test_attr = x_test_attr.reshape((x_test_attr.shape[0], x_test_attr.shape[1], 1))

    input1_shape = x_train_cgm.shape[1:]
    input2_shape = x_train_attr.shape[1:]

    if data_flag == 'cgm':
        classifier = approach(output_dir,input1_shape,num_classes,verbose=True)
        classifier.fit(x_train_cgm, y_train, x_test_cgm, y_test, y_true)
    elif data_flag == 'biomarker':
        classifier = approach(output_dir,input2_shape,num_classes,verbose=True)
        classifier.fit(x_train_attr, y_train, x_test_attr, y_test, y_true)
    else:
        classifier = approach(output_dir,input1_shape,input2_shape,num_classes,verbose=True,flag_att=flag_att,flag_CBAM=flag_cbam)
        classifier.fit([x_train_cgm, x_train_attr], y_train, [x_test_cgm, x_test_attr], y_test, y_true)

    res = classifier.evaluate()
    return res

def run_experiment(x_train_cgm, x_train_attr, x_test_cgm, x_test_attr, y_train, y_test, appr, flag_cbam, flag_att, output_dir, repeats=5, data_flag='both'):
    scores = list()
    for r in range(repeats):
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        if os.path.exists(output_dir + '/repeat' + str(r) + r'/') is False:
            os.mkdir(output_dir + '/repeat' + str(r) + r'/')
        output_dir1 = output_dir + '/repeat' + str(r) + r'/'
        score = evaluate_model(x_train_cgm, x_train_attr, x_test_cgm, \
            x_test_attr, y_train, y_test, appr, flag_cbam, flag_att, output_dir1, data_flag)
        score = score * 100.0
        scores.append(score)
    return scores

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')

    data_dir = r'./data-new-2_types/'
    method_name = r'lstmCNN'
    appr = LSTM_CNN.Classifier_LSTMCNN

    group_text = 1
    num_repeat = 3

    flag_cbam = 1
    flag_att = 1

    if flag_cbam==1 and flag_att==1:
        scores1 = []
        for i in range(5):
            filename1 = r'synthetic_detrended_6d_2types_f' + str(i) + r'_train.csv'
            filename2 = r'synthetic_detrended_6d_2types_f' + str(i) + r'_test.csv'
            data_type = 'both' # cgm,biomarker,both
            method_name = method_name + r'_dual_att-' + str(group_text) + '-' + str(i)
            output_dir = r'./results/' + method_name
            x_train_cgm, x_train_attr, x_test_cgm, x_test_attr, y_train, y_test = \
                load_dataset2(data_dir,filename1,filename2,flag_addNoise=True)
            scores = run_experiment(x_train_cgm, x_train_attr, x_test_cgm, x_test_attr, \
                y_train, y_test, appr, flag_cbam, flag_att, output_dir=output_dir, repeats=num_repeat, data_flag=data_type)
            scores1.extend(scores)

        for r in range(num_repeat*5):
            print('>#%d: %.3f' % (r+1, scores1[r]))

        m, s = np.mean(scores1), np.std(scores1)
        print('Accuracy: %.3f%% (+/- %.3f)' % (m, s))