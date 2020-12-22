# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
sys.path.append("/data/LeiLixiang/pycharmproject/DeepCTR-Torch/")
print(sys.path)
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import sys
print(sys.path)

def read_part_data(part_name):
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    features = dense_features + sparse_features

    pkl_file = open('../data/features_min.pkl', 'rb')
    features_min = pickle.load(pkl_file)

    x_npy = np.load('/data/LeiLixiang/pycharmproject/Fi_GNN/data/Criteo/'+part_name+'/train_x2.npy')
    x = pd.DataFrame(x_npy, columns=features)
    index_npy = np.load('/data/LeiLixiang/pycharmproject/Fi_GNN/data/Criteo/'+part_name+'/train_i.npy')
    index = pd.DataFrame(index_npy, columns=features)
    y_npy = np.load('/data/LeiLixiang/pycharmproject/Fi_GNN/data/Criteo/'+part_name+'/train_y.npy')
    y = pd.DataFrame(y_npy, columns=['label'])

    x = pd.concat([x[dense_features], index[sparse_features]], axis=1)

    for feat in sparse_features:
        x[feat] -= features_min[feat]

    return pd.concat([x, y], axis=1)

if __name__ == "__main__":

    pkl_file = open('../data/features_num.pkl', 'rb')
    features_num = pickle.load(pkl_file)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    features = dense_features + sparse_features

    target = ['label']
    
    fixlen_feature_columns = [SparseFeat(feat, features_num[feat]+1) for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # 3.generate input data for model

    # 4.Define Model,train,predict and evaluate


    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = AutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    model.load_state_dict(torch.load("model_autoint/model_epoch_0.pth"))
    print(model.embedding_dict['C1'](torch.tensor(0).to(device)))

    valid = read_part_data('part2')
    valid_model_input = {name: valid[name] for name in feature_names}
    for epoch in range(1,3):
        for i in range(3, 11):
            train = read_part_data('part'+str(i))
            train_model_input = {name: train[name] for name in feature_names}
            model.fit_one_epoch(train_model_input, train[target].values, shuffle=False,batch_size=1024, verbose=1, epoch=epoch)
            
            valid_ans = model.predict(valid_model_input, 102400, epoch)
            print("valid LogLoss", round(log_loss(valid[target].values, valid_ans), 4))
            print("valid AUC", round(roc_auc_score(valid[target].values, valid_ans), 4))
            print("part"+str(i)+" have finished")
        torch.save(obj=model.state_dict(), f="model_autoint/model_epoch_{}.pth".format(str(epoch)))
    test = read_part_data('part1')
    test_model_input = {name: test[name] for name in feature_names}
    pred_ans = model.predict(test_model_input, 1024, 2)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
