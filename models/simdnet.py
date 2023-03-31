import torch
from torch import nn
from torch.nn import Linear
# from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from math import sqrt
import torch.nn.functional as F
from torch_sparse import SparseTensor
from math import pi as PI
from dig.threedgraph.method.dimenetpp.dimenetpp import emb,ResidualLayer
from models.mol_features import Feature



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# edge_dims=[2,2,2,8,3]


def swish(x):
    return x*x.sigmoid()



def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False):

    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    # Calculate torsions.
    if use_torsion:
        idx_batch = torch.arange(len(idx_i), device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets - 1
        num_triplets_t = num_triplets.repeat_interleave(repeat)
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], \
                                                          idx_batch_t[mask]

        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        torsion1 = torch.atan2(b, a)  # -pi to pi
        torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
        torsion = scatter(torsion1, idx_batch_t, reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji

    else:
        return dist, angle, i, j, idx_kj, idx_ji







class EmbeddingLayer(nn.Module):
    def __init__(self, features_num_list, emb):
        super(EmbeddingLayer, self).__init__()
        self.EmbList = nn.ModuleList()
        self.num_layers = len(features_num_list)
        self.num_emb = emb
        for i, word_num in enumerate(features_num_list):
            emb_tmp = nn.Embedding(word_num, self.num_emb)
            nn.init.xavier_uniform_(emb_tmp.weight.data)
            self.EmbList.append(emb_tmp)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.EmbList[i].weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, v_f):

        g_tmp = 0

        total_dim = v_f.shape[1]
        for i in range(self.num_layers):
            g_tmp += self.EmbList[i](v_f[:, i].long())

        _, res = torch.split(v_f, [i + 1, total_dim - i - 1], dim=1)

        return torch.cat((g_tmp, res), dim=1)




class EdgeEmbedding(nn.Module):
    def __init__(self, features_num_list, emb):
        super(EdgeEmbedding, self).__init__()
        self.EmbList = nn.ModuleList()
        self.num_layers = len(features_num_list)
        self.num_emb = emb
        for i, word_num in enumerate(features_num_list):
            emb_tmp = nn.Embedding(word_num, self.num_emb)
            nn.init.xavier_uniform_(emb_tmp.weight.data)
            self.EmbList.append(emb_tmp)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.EmbList[i].weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, b_f):

        e_tmp = 0

        for i in range(self.num_layers):
            e_tmp += self.EmbList[i](b_f[:, i].long())

        return e_tmp

class init_v1(torch.nn.Module):
    def __init__(self, feature_list, num_radial, hidden_channels, act=swish):
        super(init_v1, self).__init__()
        self.act = act

        self.emb = EmbeddingLayer(feature_list, hidden_channels - 1)

        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):

        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()

        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _ = emb

        x = self.emb(x)

        rbf0 = self.act(self.lin_rbf_0(rbf))

        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))

        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2




class update_v_v1(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v_v1, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, num_nodes):
        _, e2 = e
        v = scatter(e2, i, dim=0, dim_size=num_nodes)  #
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)

        return v



def get_edge_avg(e, padding):
    _, e2 = e
    pad = padding.squeeze(-1)
    e_f = scatter(e2, pad, dim=0, reduce='mean')
    return e_f[:-1, :]



def get_node_out(v, padding):
    p = padding.squeeze(1)
    tmp0 = scatter(v, p, dim=0, reduce='mean')
    tmp1 = scatter(v, p, dim=0, reduce='max')
    tmp = torch.cat((tmp0[:-1], tmp1[:-1]), 1)
    return tmp



class update_e_v2_noself(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial,
                 num_before_skip, num_after_skip, act=swish):
        super(update_e_v2_noself, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf = emb
        x1, _ = x

        # x_ji = self.act(self.lin_ji(x1))####
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_kj  ####x_ji +
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1))  ##+ x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2








class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / sqrt(self.hidden_size),
                                1.0 / sqrt(self.hidden_size))

    def forward(self, node, batch):

        batch_nodes_num = get_batch_nodes_num(batch)
        hidden = scatter(node, batch, dim=0, reduce='max').unsqueeze(0).repeat(2, 1, 1)  # 1,
        message = self.relu(node + self.bias)
        max_node_num = torch.tensor(batch_nodes_num).max()

        ##padding

        message_list = torch.split(message, batch_nodes_num)
        padding_message = []
        for i, message_sub in enumerate(message_list):
            padding_message.append(
                F.pad(message_sub, (0, 0, 0, max_node_num - message_sub.shape[0]), "constant", 0).unsqueeze(0))

        padding_message = torch.cat(padding_message, dim=0)
        gru_message, hidden = self.gru(padding_message, hidden)

        ##

        ori_message_list = []

        for i, sub_num in enumerate(batch_nodes_num):
            ori_message_list.append(gru_message[i, :sub_num].view(-1, 2 * self.hidden_size))

        current_message = torch.cat(ori_message_list, dim=0)

        return current_message


def get_batch_nodes_num(batch):
    batch = batch.tolist()
    nodes_num = 1
    batch_nodes_num = []
    for i in range(len(batch) - 1):
        if batch[i] == batch[i] == batch[i + 1]:
            nodes_num += 1
        else:
            batch_nodes_num.append(nodes_num)
            nodes_num = 1
    batch_nodes_num.append(nodes_num)

    return batch_nodes_num


class SimdBlock(torch.nn.Module):

    def __init__(
            self, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=128, int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal',
            feature_list=None):
        super(SimdBlock, self).__init__()

        self.cutoff = cutoff


        self.init_e = init_v1(feature_list, num_radial, hidden_channels, act)
        self.init_v = update_v_v1(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v_v1(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in
            range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e_v2_noself(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])


        self.gru = BatchGRU(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch, edge_index, padding = batch_data.v_f, batch_data.pos, batch_data.batch, batch_data.edge_index, batch_data.atom_padding

        num_nodes = z.size(0)
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False)

        emb = self.emb(dist, angle, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)  ########
        v = self.init_v(e, i, num_nodes)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i, num_nodes) + v

        v = self.gru(v, batch)

        out = get_node_out(v, padding)

        return out



class Simd(nn.Module):
    def __init__(self, hidden_channel_out=512, out_dim1=64,
                 cutoff=5.0, num_layers=4,
                 hidden_channels=128, out_channels=128, int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_output_layers=3,
                 act=swish, output_init='GlorotOrthogonal',

                 ):
        super(Simd, self).__init__()
        self.graph_net = SimdBlock(cutoff=cutoff,num_layers=num_layers,
                 hidden_channels=hidden_channels, out_channels=out_channels, int_emb_size=int_emb_size, basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels,
                 num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent,
                 num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_output_layers=num_output_layers,
                 act=act, output_init=output_init,feature_list=[2]+Feature.feature_dims)
        self.lin0 = nn.Linear(hidden_channel_out, out_dim1)
        self.act = swish
        self.lin1 = nn.Linear(out_dim1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin0.weight, scale=2.0)
        self.lin0.bias.data.fill_(0)
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)

    def forward(self, gc):
        gc_v = self.graph_net(gc)

        out = self.lin1(self.act(self.lin0(gc_v)))


        return out.squeeze(1)
