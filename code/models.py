import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from mamba_ssm.models.mixer_seq_simple import MixerModel
from kan import KANLinear
from einops import reduce
import math

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def contrastive_loss(smiles_emb, graph_emb, temperature=0.01, hard_negative=False):
    # 1. 计算余弦相似度
    sim_matrix = F.cosine_similarity(smiles_emb.unsqueeze(1), graph_emb.unsqueeze(0), dim=-1)

    # 2. 提取正负样本
    labels = torch.arange(smiles_emb.size(0)).to(smiles_emb.device)

    # 3. 硬负样本挖掘（可选）
    if hard_negative:
        # 对负样本进行筛选，保留相似度较高的负样本
        neg_mask = sim_matrix < sim_matrix.max(dim=1, keepdim=True)[0]  # 找出所有非正样本
        hard_neg_sim_matrix = sim_matrix * neg_mask.float()  # 只保留“硬负样本”
        sim_matrix = torch.cat([sim_matrix, hard_neg_sim_matrix], dim=1)  # 组合正样本和硬负样本

    # 4. 归一化处理：使用温度缩放调整不同模态之间的相似度
    sim_matrix = sim_matrix / temperature


    loss = F.cross_entropy(sim_matrix, labels)

    return loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


class IEGMCL(nn.Module):
    def __init__(self, **config):
        super(IEGMCL, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_layers = config["DRUG"]["LAYERS"]
        drug_num_head = config["DRUG"]["NUM_HEAD"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_layers = config["PROTEIN"]["LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        protein_padding = config["PROTEIN"]["PADDING"]

        mgn_emb_dim = config["MGN"]["EMBEDDING_DIM"]

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.smiles_extractor  = MixerModel(d_model=drug_embedding,n_layer=drug_layers,d_intermediate=drug_num_head,vocab_size=65,rms_norm=True)
        self.protein_extractor  = MixerModel(d_model=protein_emb_dim,n_layer=protein_layers,d_intermediate=protein_num_head,vocab_size=26,rms_norm=True)
        #Multimodal Gating Network
        self.multi_gating_network = MultimodalGatingNetwork(mgn_emb_dim)
        #MLPDecoder
        self.mlp_classifier = KANClassifier(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, smi_d, bg_d, v_p, mode="train"):
        #Drug Encoder
        v_d = self.drug_extractor(bg_d)
        v_s = self.smiles_extractor(smi_d.long())
        #Protein Encoder
        v_p = self.protein_extractor(v_p.long())
        #Multimodal Gating Network
        f, v_d, v_s, v_p = self.multi_gating_network(v_d, v_s, v_p)
        #Decoder
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_s, v_p, f, score
        elif mode == "eval":
            return v_d, v_s, v_p, score, None

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y

def compute_entropy(features):
    # 先对特征进行归一化
    prob = F.softmax(features, dim=-1)
    
    # 计算信息熵
    entropy = - torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
    return entropy

class MultimodalGatingNetwork(nn.Module):
    def __init__(self, dim):
        super(MultimodalGatingNetwork, self).__init__()
        self.gated_g = GLU(dim, dim) 
        self.gated_s = GLU(dim, dim)
        self.gated_p = GLU(dim, dim)
        self.tanh = nn.Tanh()

    def forward(self, mg, ms, mp):
        mg = self.gated_g(mg) + mg
        ms = self.gated_s(ms) + ms
        mp = self.gated_p(mp) + mp
        # print(mg.shape,ms.shape,mp.shape,111111111111111111111111111111)
        v_d = reduce(mg, "b h w -> b w", 'max')
        v_s = reduce(ms, "b h w -> b w", 'max')
        v_p = reduce(mp, "b h w -> b w", 'max')
        # print(v_d.shape,v_s.shape,v_p.shape,22222222222222222222222222)
        entropy_d = compute_entropy(v_d)
        entropy_s = compute_entropy(v_s)
        total_entropy = entropy_d + entropy_s
        weight_d = (1 - entropy_d / total_entropy).unsqueeze(1)
        weight_s = (1 - entropy_s / total_entropy).unsqueeze(1)
        v_dp = weight_d * v_d * v_p
        v_sp = weight_s * v_s * v_p
        fused_feat = self.tanh(torch.cat([v_dp, v_sp], dim=-1))
        return fused_feat, v_d, v_s, v_p

class KANClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(KANClassifier, self).__init__()
        self.fc1 = KANLinear(in_features = in_dim, out_features = hidden_dim,
        grid_size=3,spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],)

        self.fc2 = KANLinear(in_features = hidden_dim, out_features =out_dim,
        grid_size=3,spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],)

        self.fc3 = KANLinear(in_features = hidden_dim, out_features =out_dim,
        grid_size=5,spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],)

        self.fc4 = KANLinear(in_features = out_dim, out_features =binary,
        grid_size=3,spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],)


        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):  # x.shpae[64, 256]

        x = self.bn1(F.relu(self.fc1(x)))

        x = self.bn2(F.relu(self.fc2(x)))
        # x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x




