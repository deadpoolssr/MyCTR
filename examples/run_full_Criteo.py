# -*- coding: utf-8 -*-
import pandas as pd
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
if __name__ == "__main__":
   
    sparse_features = ['C' + str(i) for i in range(1, 40)]
    # dense_features = []
    train = pd.read_csv('/data/LeiLixiang/pycharmproject/Fi_GNN/data/train.txt', sep = '	', header=None, names=['label']+sparse_features)
    test = pd.read_csv('/data/LeiLixiang/pycharmproject/Fi_GNN/data/test.txt', sep = '	', header=None, names=sparse_features)
    
    print("Data read finish...")

    train_x = train[sparse_features]
    test_x = test

    num_features = ['C'+ str(i) for i in range(1, 14)]
    category_features = ['C'+ str(i) for i in range(14, 40)]

    train_x[category_features] = train_x[category_features].fillna(0, )
    test_x[category_features] = test_x[category_features].fillna(0, )
    train_x[num_features] = train_x[num_features].fillna('-1', )
    test_x[num_features] = test_x[num_features].fillna('-1', )
    train_x[num_features].astype(int)
    test_x[num_features].astype(int)
    
    target = ['label']

    print("Fill in the missing data...")

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    print("Data encoding")

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = ['cuda:2','cuda:5','cuda:6']

    model = MyAutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    model.fit(train_model_input, train[target].values, batch_size=32, epochs=3, verbose=2, validation_split=0.1)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
