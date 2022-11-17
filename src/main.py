from torch_geometric.datasets import Planetoid
from train import train
from util import create_gcn_matrix

if __name__ == "__main__":
    Epoch = 100
    dataset = Planetoid(root="/tmp/Cora", name="Cora") #データセットのロード
    matrix = create_gcn_matrix(data=dataset[0])
    train(dataset=dataset, matrix=matrix, Epoch=Epoch)