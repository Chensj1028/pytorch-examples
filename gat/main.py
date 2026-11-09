import os
import time
import requests
import tarfile
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


################################
###  GAT LAYER DEFINITION    ###
# GAT 层定义。
################################

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    # 论文 Graph Attention Networks 中的图注意力层。
    # 先计算 e_ij = a(W h_i, W h_j)，再用按邻居归一化的 softmax 得到 α_ij，
    # 最后计算 h_i' = σ(Σ_j(α_ij W h_j))。
    # h_i、h_j 分别为节点 i、j 的特征；W 为可学习矩阵；a 为注意力机制；σ 为激活函数。
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        # 注意力头数量。
        self.concat = concat # wether to concatenate the final attention heads
        # 是否拼接最后的各注意力头。
        self.dropout = dropout # Dropout rate
        # Dropout 比例。

        if concat: # concatenating the attention heads
        # 拼接各注意力头。
            self.out_features = out_features # Number of output features per node
            # 每个节点的输出特征数。
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            # 确保 out_features 是 n_heads 的整数倍。
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
        # 对各注意力头的输出取平均，原论文采用此方式。
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        # 对每个节点应用由权重矩阵 W 参数化的共享线性变换。
        #  Initialize the weight matrix W 
        # 初始化权重矩阵 W。
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

        # Initialize the attention weights a
        # 初始化注意力权重 a。
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        # LeakyReLU 激活函数。
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients
        # 对注意力系数应用 softmax 激活函数。

        self.reset_parameters() # Reset the parameters
        # 重置参数。


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        # 重新初始化可学习参数。
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        # 以向量化并行方式计算图中所有节点对 (i, j) 的注意力分数：
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])；其中 || 表示拼接，a 和 W 为可学习参数。
        # 参数 h_transformed：变换后的特征，文档给出的形状为 (n_nodes, n_heads, n_hidden)。
        # n_nodes 为节点数，out_features 为每个节点的输出特征数。
        # 返回值：形状为 (n_heads, n_nodes, n_nodes) 的注意力分数矩阵。
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add 
        # 广播相加。
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        # 形状广播：(头数, 节点数, 1) + (头数, 1, 节点数) = (头数, 节点数, 节点数)。
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        # 执行图注意力层运算。
        # 参数 h：节点特征张量；adj_mat：表示图结构的邻接矩阵。
        # 返回值：图注意力运算后的输出张量。
        n_nodes = h.shape[0]

        # Apply linear transformation to node feature -> W h
        # 对节点特征应用线性变换，得到 W h。
        # output shape (n_nodes, n_hidden * n_heads)
        # 输出形状：（节点数，隐藏特征数×头数）。
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # 通过调整张量形状拆分各头，并将头维度放在首位。
        # output shape (n_heads, n_nodes, n_hidden)
        # 输出形状：（头数，节点数，隐藏特征数）。
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        # getting the attention scores
        # 计算注意力分数。
        # output shape (n_heads, n_nodes, n_nodes)
        # 输出形状：（头数，节点数，节点数）。
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        # 将不存在边的位置的注意力分数设为 -9e15，以屏蔽这些边。
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        # 掩码处理后的注意力分数。
        
        # attention coefficients are computed as a softmax over the rows
        # 沿每一行进行 softmax，计算注意力系数，
        # for each column j in the attention score matrix e
        # 对应注意力分数矩阵 e 中的各列 j。
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        # 通过对邻居特征加权平均，得到最终节点嵌入。
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # 拼接各注意力头或对它们取平均。
        # output shape (n_nodes, out_features)
        # 输出形状：（节点数，输出特征数）。
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime

################################
### MAIN GAT NETWORK MODULE  ###
# GAT 主网络模块。
################################

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    # 论文 Graph Attention Networks 中的图注意力网络，由两层图注意力层组成。
    # 第一层后接 ELU；第二层为单注意力头，并使用 softmax；本实现最终输出使用 log-softmax。
    def __init__(self,
        in_features,
        n_hidden,
        n_heads,
        num_classes,
        concat=False,
        dropout=0.4,
        leaky_relu_slope=0.2):
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """
        # 初始化 GAT 模型。
        # 参数 in_features：每个节点的输入特征数；n_hidden：第一层输出大小；n_heads：第一层注意力头数。
        # num_classes：节点预测类别数；concat：第一层输出是否拼接各头，否则取平均，默认 False。
        # dropout：丢弃比例，默认 0.4；leaky_relu_slope：LeakyReLU 负半轴斜率，默认 0.2。

        super(GAT, self).__init__()

        # Define the Graph Attention layers
        # 定义图注意力层。
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        # 执行网络前向传播。
        # 参数 input_tensor：节点特征张量；adj_mat：表示图结构的邻接矩阵。
        # 返回值：前向传播的输出张量。

        # Apply the first Graph Attention layer
        # 应用第一个图注意力层。
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer
        # 对第一层输出应用 ELU 激活函数。

        # Apply the second Graph Attention layer
        # 应用第二个图注意力层。
        x = self.gat2(x, adj_mat)

        return F.log_softmax(x, dim=1) # Apply log softmax activation function
        # 应用 log-softmax 激活函数。

################################
### LOADING THE CORA DATASET ###
# 加载 Cora 数据集。
################################

def load_cora(path='./cora', device='cpu'):
    """
    Loads the Cora dataset. The dataset is downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

    """
    # 加载 Cora 数据集，下载地址见上方链接。

    # Set the paths to the data files
    # 设置数据文件路径。
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    # 从文件加载数据。
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    # 处理特征。
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    # 提取特征值。
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    # 计算每个节点的特征之和。
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    # 计算各特征和的倒数。
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    # 处理除零情况。
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    # 将缩放向量转换为稀疏对角矩阵。
    features = scale_vector @ features # Scale the features using the scale vector
    # 使用缩放向量缩放特征。

    # Process labels
    # 处理标签。
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    # 提取不同类别，并将标签映射为索引。
    labels = torch.LongTensor(labels) # Convert labels to a tensor
    # 将标签转换为张量。

    # Process adjacency matrix
    # 处理邻接矩阵。
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    # 提取节点索引。
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions
    # 创建从索引到位置的映射字典。

    # Map node indices to positions in the adjacency matrix
    # 将节点索引映射到邻接矩阵中的位置。
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    # 节点数量。
    E = edges.shape[0] # Number of edges
    # 边数量。
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    # 以稀疏张量创建初始邻接矩阵。
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix
    # 向邻接矩阵添加自环。

    # return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)
    return features.to(device), labels.to(device), adj_mat.to(device)

#################################
### TRAIN AND TEST FUNCTIONS  ###
# 训练与测试函数。
#################################

def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    # Forward pass
    # 前向传播。
    output = model(*input) 
    loss = criterion(output[mask_train], target[mask_train]) # Compute the loss using the training mask
    # 使用训练掩码计算损失。

    loss.backward()
    optimizer.step()

    # Evaluate the model performance on training and validation sets
    # 在训练集和验证集上评估模型性能。
    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        # Print the training progress at specified intervals
        # 按指定间隔打印训练进度。
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)
    return loss.item(), acc.item()


if __name__ == '__main__':

    # Training settings
    # 训练设置。
    # All defalut values are the same as in the config used in the main paper
    # 所有默认值均与原论文使用的配置一致。

    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay (default: 6e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6,
                        help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of the attention heads (default: 4)')
    parser.add_argument('--concat-heads', action='store_true',
                        help='wether to concatinate attention heads, or average over them (default: False)')
    parser.add_argument('--val-every', type=int, default=20,
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', 
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S',
                        help='random seed (default: 13)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()

    # Set the device to run on
    # 设置运行设备。
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')
    print(f'Using {device} device')

    # Load the dataset
    # 加载数据集。
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    path = './cora'

    if os.path.isfile(os.path.join(path, 'cora.content')) and os.path.isfile(os.path.join(path, 'cora.cites')):
        print('Dataset already downloaded...')
    else:
        print('Downloading dataset...')
        with requests.get(cora_url, stream=True) as tgz_file:
            with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
                tgz_object.extractall()

    print('Loading dataset...')
    # Load the dataset
    # 加载数据集。
    features, labels, adj_mat = load_cora(device=device)
    # Split the dataset into training, validation, and test sets
    # 将数据集划分为训练集、验证集和测试集。
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1200], idx[1200:1600], idx[1600:]


    # Create the model
    # 创建模型。
    # The model consists of a 2-layer stack of Graph Attention Layers (GATs).
    # 模型由两层图注意力层（GAT）堆叠组成。
    gat_net = GAT(
        in_features=features.shape[1],          # Number of input features per node  
        # 每个节点的输入特征数。
        n_hidden=args.hidden_dim,               # Output size of the first Graph Attention Layer
        # 第一个图注意力层的输出大小。
        n_heads=args.num_heads,                 # Number of attention heads in the first Graph Attention Layer
        # 第一个图注意力层的注意力头数。
        num_classes=labels.max().item() + 1,    # Number of classes to predict for each node
        # 为每个节点预测的类别数量。
        concat=args.concat_heads,               # Wether to concatinate attention heads
        # 是否拼接各注意力头。
        dropout=args.dropout_p,                 # Dropout rate
        # Dropout 比例。
        leaky_relu_slope=0.2                    # Alpha (slope) of the leaky relu activation
        # LeakyReLU 激活函数的 alpha，即负半轴斜率。
    ).to(device)

    # configure the optimizer and loss function
    # 配置优化器和损失函数。
    optimizer = Adam(gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    # Train and evaluate the model
    # 训练并评估模型。
    for epoch in range(args.epochs):
        train_iter(epoch + 1, gat_net, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
        if args.dry_run:
            break
    loss_test, acc_test = test(gat_net, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')