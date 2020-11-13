import torch
import torch.nn as nn
import scipy.sparse as sp
from torch.nn.functional import softmax
from annoy import AnnoyIndex

class KNNAttentionLayer(nn.Module):
    def __init__(self, K, embedding_dict, feature_columns,feature_index):
        super(KNNAttentionLayer, self).__init__()
        self.K = K
        self.embedding_dict = embedding_dict
        self.feature_adj = {}
        self.feature_index = feature_index
        self.feature_columns = feature_columns
        for feat in feature_columns:
            adj = []
            t = AnnoyIndex(feat.embedding_dim, 'angular')
            embeddings = embedding_dict[feat.embedding_name]
            for i in range(embeddings.num_embeddings):
                t.add_item(i, embeddings(torch.tensor(i).cuda()).cpu().detach().numpy())
            t.build(1000)
            for i in range(embeddings.num_embeddings):
                index_list = t.get_nns_by_item(i, self.K+1)
                index_list.remove(i)
                if len(index_list)>0:
                    adj.append(index_list)
            self.feature_adj[feat.embedding_name] = torch.as_tensor(adj).cuda()
        self.GATList = {feat.embedding_name : GAT(self.feature_adj[feat.embedding_name],
                     embedding_dict[feat.embedding_name], self.K) if self.feature_adj[feat.embedding_name].size()[0]>0 
                     else None for feat in feature_columns}

    def forward(self, X):
        output = [self.GATList[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
                    if self.GATList[feat.embedding_name] != None else 
                    embedding_dict[feat](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
                         for feat in self.feature_columns]
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
        # print(edge.size())
        # print(batch_embeddings.size())
        # print(neighbor_embeddings.size())
        weights = softmax(torch.matmul(batch_embeddings, neighbor_embeddings.permute(0,2,1)), dim=-1)
        result = torch.matmul(weights, neighbor_embeddings) + batch_embeddings
        return result