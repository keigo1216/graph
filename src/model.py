import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GCNConv(torch.nn.Module):
    def __init__(self, num_node_in_features, num_node_out_features, matrix):
        """
        Parameters
        --------------
        num_node_in_features: int
            入力の特徴量の次元
        num_node_out_features: int
            出力の特徴量の次元
        matrix: torch.Tensor (num_nodes, num_nodes)
            入力データに作用させる行列
        """
        super(GCNConv, self).__init__()
        self.matrix = matrix
        self.linear = torch.nn.Linear(
            in_features=num_node_in_features,
            out_features=num_node_out_features,
            bias=False
        )
    
    def forward(self, x):
        x = self.linear(torch.matmul(self.matrix, x)) 
        # x = self.linear(torch.matmul(self.identity_matrix - ((self.graph_laplacian / 2)**2)/2, x)) # cosをテーラー展開して作ったフィルタ、精度は40%前後
        # x = self.linear(torch.matmul(-(self.graph_laplacian-1)**2+self.identity_matrix, x)) #全く精度上がらず
    
        return x

class Net(torch.nn.Module):
    def __init__(self, matrix, num_node_features, num_classes):
        """
        Parameters
        --------------
        matrix: torch.Tensor (num_nodes, num_nodes)
            入力データに作用させる行列
        num_node_features: int
            データセットの各ノードの特徴量
        num_classes: int
            分類するノードのクラス数
        """
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16, matrix)
        self.dropout = torch.nn.Dropout(0.3)
        self.conv2 = GCNConv(16, num_classes, matrix)

    def forward(self, data):
        x = data.x
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.dropout(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
