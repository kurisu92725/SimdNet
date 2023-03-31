import os
from tqdm import tqdm
import pickle
import torch
from rdkit import Chem

from scipy.spatial import distance_matrix
from mol_features import Feature


datafiles = ['../../PLA_data/general-set-except-refined', '../../PLA_data/refined-set', '../../PLA_data/coreset','../../PLA_data/v2013-core']
labelfiles = datafiles[0] + '/index/INDEX_general_PL_data.2016'



def load_pk_data(data_path):
    res = dict()
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    return res


def build_complex_graph(ligand_pos, pocket_pos, ligand_feature, pocket_feature, lb_begin, lb_end, lb_feature, pb_begin,
                        pb_end, pb_feature):
    ##args:
    intra_cutoff = 3.0
    inter_cutoff = 5.0

    ligand_len = len(ligand_pos)
    pocket_len = len(pocket_pos)
    bond_feature_len = len(lb_feature[0])

    l_p_tensor = (torch.from_numpy(ligand_pos))
    p_p_tensor = (torch.from_numpy(pocket_pos))

    dm_l=distance_matrix(ligand_pos,ligand_pos)
    dm_p=distance_matrix(pocket_pos,pocket_pos)
    dm_lp = distance_matrix(ligand_pos, pocket_pos)

    ligand_begin=[]
    ligand_end=[]
    pocket_begin=[]
    pocket_end=[]

    for i in range(ligand_len):
        for j in range(ligand_len):
            if 0.1<dm_l[i,j]<intra_cutoff:
                ligand_begin.append(i)
                ligand_end.append(j)

    for i in range(pocket_len):
        for j in range(pocket_len):
            if 0.1 < dm_p[i,j] < intra_cutoff:
                pocket_begin.append(i)
                pocket_end.append(j)

    edge_index_ligand=torch.tensor([ligand_begin,ligand_end],dtype=torch.long)
    edge_from_ligand=torch.zeros((edge_index_ligand.shape[1],1),dtype=torch.long)

    edge_index_pocket=torch.tensor([pocket_begin,pocket_end],dtype=torch.long)
    edge_from_pocket=torch.ones((edge_index_pocket.shape[1],1),dtype=torch.long)

    complex_feature = torch.cat((torch.tensor(ligand_feature), torch.tensor(pocket_feature)),
                                dim=0)  ##
    complex_pos = torch.cat((l_p_tensor, p_p_tensor),
                            dim=0)  #
    ##
    ##edge_inf
    lb_feature.append([0] * bond_feature_len)
    pb_feature.append([0] * bond_feature_len)
    lb_feature = torch.tensor(lb_feature)
    pb_feature = torch.tensor(pb_feature)

    tmp_index_l = torch.zeros((ligand_len, ligand_len), dtype=torch.long).fill_(-1)
    tmp_index_p = torch.zeros((pocket_len, pocket_len), dtype=torch.long).fill_(-1)
    tmp_index_l[lb_begin, lb_end] = torch.arange(len(lb_begin))
    tmp_index_p[pb_begin, pb_end] = torch.arange(len(pb_begin))
    l_b_f = lb_feature[tmp_index_l[edge_index_ligand[0].tolist(), edge_index_ligand[1].tolist()]]
    p_b_f = pb_feature[tmp_index_p[edge_index_pocket[0].tolist(), edge_index_pocket[1].tolist()]]  ##

    edge_index_pocket = edge_index_pocket + ligand_len  ##

    complex_add_index_b = []
    complex_add_index_e = []
    tmp_add_f = []
    for i in range(ligand_len):
        for j in range(pocket_len):
            if dm_lp[i, j] < inter_cutoff:
                complex_add_index_b.append(i)
                complex_add_index_b.append(j + ligand_len)
                complex_add_index_e.append(j + ligand_len)
                complex_add_index_e.append(i)
                tmp_add_f.append([0] * bond_feature_len)
                tmp_add_f.append([0] * bond_feature_len)

    complex_add_index = torch.tensor([complex_add_index_b, complex_add_index_e], dtype=torch.long)
    edge_from_mid=torch.zeros((complex_add_index.shape[1],1)).fill_(2).long()
    tmp_add_f = torch.tensor(tmp_add_f)

    inter_num=tmp_add_f.shape[0]

    complex_index = torch.cat((edge_index_ligand, edge_index_pocket, complex_add_index),
                              dim=1)  #
    complex_bond_feature = torch.cat((l_b_f, p_b_f, tmp_add_f), dim=0)

    complex_edge_from=torch.cat((edge_from_ligand,edge_from_pocket,edge_from_mid),dim=0)

    complex_bond_feature=torch.cat((complex_bond_feature,complex_edge_from),dim=1)



    dict = {}
    dict['ligand_num']=ligand_len
    dict['pocket_num']=pocket_len
    dict['ligand_edge_num']=l_b_f.shape[0]
    dict['pocket_edge_num']=p_b_f.shape[0]
    dict['intra_edge_num']=l_b_f.shape[0]+p_b_f.shape[0]
    dict['inter_edge_num']=tmp_add_f.shape[0]

    dict['complex_pos'] = complex_pos
    dict['complex_feature'] = complex_feature
    dict['complex_index'] = complex_index
    dict['complex_bond_f'] = complex_bond_feature


    return dict


def get_complex_dict(path, name, class_feature):
    ligand = Chem.MolFromMolFile('%s/%s/%s_ligand.mol' % (path, name, name))
    pocket = Chem.MolFromPDBFile('%s/%s/%s_pocket.pdb' % (path, name, name))

    sig = 0
    if ligand == None:
        sig = 0
    if pocket == None:
        sig = 1
    if ligand == None and pocket == None:
        sig = 2
    if ligand == None or pocket == None:
        return sig

    lig_f, lig_c = class_feature.get_features(ligand)
    lig_begin, lig_end, lig_bond_f = class_feature.get_real_bonds(ligand)
    pocket_f, pocket_c = class_feature.get_features(pocket)
    pocket_begin, pocket_end, pocket_bond_f = class_feature.get_real_bonds(pocket)

    return build_complex_graph(lig_c, pocket_c, lig_f, pocket_f, lig_begin, lig_end, lig_bond_f, pocket_begin,
                               pocket_end, pocket_bond_f)



def getlabels():
    labels = load_pk_data(labelfiles)
    pick_file = open('./pdb_2016_labels.pkl', 'wb')
    pickle.dump(labels, pick_file)
    pick_file.close()


def orifile2d():
    data_path = datafiles[3]
    save_ = '../../PLA_data_pre/data11/'
    save_name = 'v2013-core'
    save_path = save_ + save_name
    class_feature = Feature()
    name_pointer = []
    failuer_pointer = [[], [], []]
    for name in tqdm(os.listdir(data_path)):
        if len(name) != 4:
            continue

        tmp_dic = get_complex_dict(data_path, name, class_feature)
        if tmp_dic == 0 or tmp_dic == 1 or tmp_dic == 2:
            failuer_pointer[tmp_dic].append(name)
            print(f'failures happen in {name},error: {tmp_dic} ')
            continue

        tmp_name = name
        name_pointer.append(tmp_name)
        pick_file = open(save_path + '/' + tmp_name + '.pkl', 'wb')
        pickle.dump(tmp_dic, pick_file)
        pick_file.close()

        # pick_file = open(save_path + '/' + name + '.pkl', 'wb')
        # pickle.dump(tmp_dic,pick_file)
        # pick_file.close()

    pick_file = open(save_ + save_name + '_name_pointer.pkl', 'wb')
    pickle.dump(name_pointer, pick_file)
    pick_file.close()
    #
    pick_file = open(save_ + 'failure_' + save_name + '_name_pointer.pkl', 'wb')
    pickle.dump(failuer_pointer, pick_file)
    pick_file.close()


if __name__ == '__main__':
    print('***********************test*************************')
    #

    # print(a[0])

    orifile2d()
    # distance_dict()
    # get_softmax()
    # get_files_max()

    # getlabels()

    # pick_file = open('./pdb_2016_labels.pkl', 'rb')
    # a=pickle.load(pick_file)
    # print(a['1a30'])