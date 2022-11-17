import torch

def create_adj_matrix(data, renormalaize=False):
    """
    隣接行列を計算する

    Parameters
    --------------
    data: torch_geometric.data.data.Data
        データセットの配列の要素のデータ型
        data.num_nodes: ノードの数
        data.edge_index[0]: エッジの始点
        data.edge_index[1]: エッジの終点
    renormalaize: Boolen
        True: 正規化した隣接行列D^(0.5)AD^(0.5)を返す
        False: 隣接行列を返す
        
    Return
    --------------
    adj_matrix: torch.Tensor(num_nodes, num_nodes)
        隣接行列を返す
    """
    num_nodes = data.num_nodes
    edge_of_start_point = data.edge_index[0]
    edge_of_end_point = data.edge_index[1]

    adj_matrix = torch.zeros([num_nodes, num_nodes], dtype=torch.float32)
    adj_matrix[edge_of_start_point, edge_of_end_point] = 1

    if renormalaize is True:
        degree_of_node = torch.sum(adj_matrix, dim=1)
        degree_reciprocal_square_node = torch.where(degree_of_node > 0, degree_of_node**(-0.5), 0)
        degree_reciprocal_square_matrix = torch.diag(degree_reciprocal_square_node)
        adj_matrix = torch.matmul(torch.matmul(degree_reciprocal_square_matrix, adj_matrix), degree_reciprocal_square_matrix)
    
    return adj_matrix

def create_graph_laplasian_matrix(data):
    """
    隣接行列, 次数行列を求め, そこから正規化ラプラシアンを計算する

    Parameters
    --------------
    data: torch_geometric.data.data.Data
        データセットの配列の要素のデータ型
        data.num_nodes: ノードの数
        data.edge_index[0]: エッジの始点
        data.edge_index[1]: エッジの終点 
    
    Return
    --------------
    identity_matrix: (num_nodes, num_nodes)
        単位行列
    graph_laplacian: (num_nodes, num_nodes)
        正規化されたグラフラプラシアン
    """
    num_nodes = data.num_nodes
    edge_of_start_point = data.edge_index[0]
    edge_of_end_point = data.edge_index[1]

    identity_matrix = torch.eye(num_nodes)
    adj_matrix = torch.zeros([num_nodes, num_nodes], dtype=torch.float32)
    adj_matrix[edge_of_start_point, edge_of_end_point] = 1
    degree_of_node = torch.sum(adj_matrix, dim=1)
    degree_reciprocal_square_node = torch.where(degree_of_node > 0, degree_of_node**(-0.5), 0)
    degree_reciprocal_square_matrix = torch.diag(degree_reciprocal_square_node)
    graph_laplacian = identity_matrix - torch.matmul(
        torch.matmul(degree_reciprocal_square_matrix, adj_matrix), degree_reciprocal_square_matrix)
    
    return identity_matrix, graph_laplacian

def create_gcn_matrix(data):
    """
    GCNで必要な行列I+D^(0.5)AD^(0.5)を計算する関数

    Parameters
    --------------
    data: torch_geometric.data.data.Data
        データセットの配列の要素のデータ型
        data.num_nodes: ノードの数
        data.edge_index[0]: エッジの始点
        data.edge_index[1]: エッジの終点 
    
    Return
    --------------
    matrix: torch.Tensor (num_node, num_node)
        上記の行列を返す
    """
    num_nodes = data.num_nodes

    identity_matrix = torch.eye(num_nodes)
    adj_matrix = create_adj_matrix(
        data=data,
        renormalaize=True
    )
    return identity_matrix+adj_matrix