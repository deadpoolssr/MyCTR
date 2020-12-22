import torch
import torch.nn as nn
import scipy.sparse as sp
import pickle
from torch.nn.functional import softmax
from annoy import AnnoyIndex

class KNNAttentionLayer(nn.Module):
    def __init__(self, K, embedding_dict, feature_columns,feature_index, device):
        super(KNNAttentionLayer, self).__init__()
        self.K = K
        self.embedding_dict = embedding_dict
        self.feature_adj = {}
        self.feature_index = feature_index
        self.feature_columns = feature_columns

    def generateGraph(self, metric, device):
        for feat in self.feature_columns:
            if 'C' in feat.name:
                adj = []
                t = AnnoyIndex(feat.embedding_dim, metric)
                embeddings = self.embedding_dict[feat.name]
                for i in range(embeddings.num_embeddings):
                    t.add_item(i, embeddings(torch.tensor(i).to(device)).cpu().detach().numpy())
                t.build(1000)
                for i in range(embeddings.num_embeddings):
                    index_list = t.get_nns_by_item(i, self.K+1)
                    try:
                        index_list.remove(i)
                    except:
                        index_list.pop()
                    if len(index_list)>0:
                        adj.append(index_list)
                self.feature_adj[feat.name] = torch.as_tensor(adj).to(device)
                print(feat.name + '\'s graph have finished')

    def generateGNNList(self, device):
        for key in self.feature_adj:
            self.feature_adj[key] = self.feature_adj[key].to(device)
        print(self.feature_adj['C1'])
        # self.GNNList = {feat.name : GAT(self.feature_adj[feat.name],
        #              self.embedding_dict[feat.name], self.K) if feat.name in self.feature_adj and self.feature_adj[feat.name].size()[0]>0 
        #              else None for feat in self.feature_columns}
        self.GNNList = {feat.name : GCN(self.feature_adj[feat.name],
                    self.embedding_dict[feat.name], self.K) if feat.name in self.feature_adj and self.feature_adj[feat.name].size()[0]>0 
                    else None for feat in self.feature_columns}
        for feat_name in list(self.GNNList.keys()):
            if 'I' in feat_name:
                del(self.GNNList[feat_name])

    def forward(self, X):
        output = [self.GNNList[feat_name](X[:, self.feature_index[feat_name][0]:self.feature_index[feat_name][1]].long())
                    if self.GNNList[feat_name] != None else 
                    self.embedding_dict[feat_name](X[:, self.feature_index[feat_name][0]:self.feature_index[feat_name][1]].long())
                         for feat_name in self.GNNList.keys()]
        return torch.cat(output, dim=1)


class GAT(nn.Module):
    def __init__(self, adj, embeddings, K):
        super(GAT, self).__init__()
        self.adj = adj
        self.embeddings = embeddings
        self.K = K

    def forward(self, X):
        edge = self.adj[X]
        batch_embeddings = self.embeddings(X)
        neighbor_embeddings = self.embeddings(edge).squeeze(1)
        weights = torch.matmul(batch_embeddings, neighbor_embeddings.permute(0,2,1))
        weights = softmax(weights, dim=-1)
        result = torch.matmul(weights, neighbor_embeddings) + batch_embeddings
        return result


class GCN(nn.Module):
    def __init__(self, adj, embeddings, K):
        super(GCN, self).__init__()
        self.adj = adj
        self.embeddings = embeddings
        self.K = K

    def forward(self, X):
        edge = self.adj[X]
        batch_embeddings = self.embeddings(X)
        neighbor_embeddings = self.embeddings(edge).squeeze(1)
        result = torch.mean(neighbor_embeddings,dim=1,keepdim=True) + batch_embeddings
        return result