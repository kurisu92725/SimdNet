import pickle

import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch_geometric.data import Data, Batch

class PDBbind_2016_complex(Dataset):
    def __init__(self, namepointer_path, graph_path, label_path):  ##
        pick_file = open(namepointer_path, 'rb')
        self.name_pointer = pickle.load(pick_file)
        pick_file.close()

        pick_file2 = open(label_path, 'rb')
        self.labels = pickle.load(pick_file2)
        pick_file2.close()

        self.graph_path = graph_path

    def __getitem__(self, idx):
        name = self.name_pointer[idx]
        # glist, label = load_graphs(self.graph_path + '/' + name + '.bin')
        pick_file = open(self.graph_path + '/' + name + '.pkl', 'rb')
        graph_label = self.labels[name[:4]]
        tmpdict = pickle.load(pick_file)
        pick_file.close()
        tmp_padding=torch.zeros((tmpdict['intra_edge_num']),dtype=torch.long)
        tmp_padding=F.pad(tmp_padding, (0, tmpdict['inter_edge_num']), "constant", 1)##

        needed_atom=set((tmp_padding*(tmpdict['complex_index']+1)).reshape(-1).tolist())
        needed_atom.remove(0)
        needed_atom = torch.tensor(list(needed_atom))-1
        atom_padding=torch.zeros((tmpdict['complex_pos'].shape[0],1),dtype=torch.long)
        atom_padding[needed_atom]=1

        add_atom_cls=torch.zeros((tmpdict['complex_pos'].shape[0],1),dtype=torch.long)
        add_atom_cls[:tmpdict['ligand_num']]=1


        graph_complex = Data(edge_index=tmpdict['complex_index'],
                             pos=tmpdict['complex_pos'].float(),
                             v_f=torch.cat((add_atom_cls.float(),tmpdict['complex_feature'].float()),1),
                             atom_padding=atom_padding,
                             intra_edge_num=tmpdict['intra_edge_num'],
                             ligand_num=tmpdict['ligand_num'],
                             pocket_num=tmpdict['pocket_num']
                             )##

        return graph_complex, graph_label

    def __len__(self):
        return len(self.name_pointer)


def collate_6(batch_data):
    g_c = []
    labels = []
    max_len=len(batch_data)
    for i,(gc, l) in enumerate(batch_data):
        gc.atom_padding[torch.where(gc.atom_padding==0)]=max_len
        gc.atom_padding[torch.where(gc.atom_padding == 1)] = i
        g_c.append(gc)
        labels.append(l)

    graph_c = Batch.from_data_list(g_c)

    return graph_c, torch.tensor(labels)



def dataset_for_pyg(general_name,general_set,core_name,core_set,label_file):
    train_valid=PDBbind_2016_complex(general_name,general_set,label_file)
    test_dataset=PDBbind_2016_complex(core_name,core_set,label_file)
    train_size = int(len(train_valid) * 0.99)
    valid_size = len(train_valid) - train_size
    test_size= len(test_dataset)
    train_dataset=torch.utils.data.Subset(train_valid,range(train_size))
    valid_dataset=torch.utils.data.Subset(train_valid,range(train_size,len(train_valid)))

    print(f'train size:{len(train_dataset)} \n valid size:{len(valid_dataset)} \n test size:{test_size}')

    return train_dataset,valid_dataset,test_dataset

