from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Net


def train(dataset, matrix, Epoch):
    """
    モデルの学習

    Parameter
    --------------
    data: torch_geometric.data.data.Data
        データセットの配列の要素のデータ型
    """
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    data = dataset[0]

    model = Net(
        matrix=matrix,
        num_node_features=num_node_features,
        num_classes=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(Epoch):
        
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0 
        
        #モデルの学習
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        _, train_pred = out.max(dim=1)
        train_acc = float(train_pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        train_acc = train_acc / data.train_mask.sum().item()

        #モデルの推論
        with torch.no_grad():
            model.eval()
            out = model(data)
            loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
            val_loss = loss.item()
            _, val_pred = out.max(dim=1)
            val_acc = float(val_pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            val_acc = val_acc / data.test_mask.sum().item()
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print('Epoch: {}/{} | loss: {:2f} | acc: {:2f} | val_loss: {:2f} | val_acc: {:2f}' .format(epoch+1, Epoch, train_loss, train_acc, val_loss, val_acc))
    
    plt.plot(range(1, Epoch+1), train_loss_list, label='training loss')
    plt.plot(range(1, Epoch+1), val_loss_list, label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(range(1, Epoch+1), train_acc_list, label='training acc')
    plt.plot(range(1, Epoch+1), val_acc_list, label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('acc.png')