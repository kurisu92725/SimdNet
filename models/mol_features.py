import numpy as np
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


class Feature():
    allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)) + ['unk'],  #
        'possible_chirality_list': [  #
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER'
        ],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'unk'],  #
        'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'unk'],  #
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'unk'],  ##
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'unk'],  ##
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'unk'],  ##
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'unk'],  #
        'possible_hybridization_list': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'unk'  ##
        ],
        'possible_is_aromatic_list': [False, True],  ##
        'possible_is_in_ring3_list': [False, True],  #
        'possible_is_in_ring4_list': [False, True],
        'possible_is_in_ring5_list': [False, True],
        'possible_is_in_ring6_list': [False, True],
        'possible_is_in_ring7_list': [False, True],
        'possible_is_in_ring8_list': [False, True],
        'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                                 'MET',
                                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV',
                                 'MEU',
                                 'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'unk'],
        #
        'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*',
                                 'OD',
                                 'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'unk'],  #
        'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2',
                                 'CH2',
                                 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O',
                                 'OD1',
                                 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'unk'],
        #
    }

    feature_dims = list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_implicit_valence_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],###
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_numring_list'],
        allowable_features['possible_is_in_ring3_list'],
        allowable_features['possible_is_in_ring4_list'],
        allowable_features['possible_is_in_ring5_list'],
        allowable_features['possible_is_in_ring6_list'],
        allowable_features['possible_is_in_ring7_list'],
        allowable_features['possible_is_in_ring8_list'],
    ])) ##

    bond_features={
        'is_aromoatic':[False,True],
        'is_conjugated':[False,True],
        'is_in_ring':[False,True],
        'type':[0,2,3,4,6,8,10,'unk']
        ##
    }#+getbondtypeasdouble


    def __init__(self):
        super(Feature, self).__init__()

    def get_features_index(self,l,e):
        try:
            return l.index(e)
        except:
            return len(l) - 1

    def get_features(self,mol):##
        ComputeGasteigerCharges(mol)  #
        ringinfo = mol.GetRingInfo()  ##

        atom_features_list = []  ##
        atom_coor_list=mol.GetConformer().GetPositions()
        for idx, atom in enumerate(mol.GetAtoms()):  ##
            g_charge = atom.GetDoubleProp('_GasteigerCharge')  ##
            atom_features_list.append([  ##
                self.get_features_index(Feature.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),#0
                Feature.allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),  ##1
                self.get_features_index(Feature.allowable_features['possible_degree_list'], atom.GetTotalDegree()),  ##2
                self.get_features_index(Feature.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),  ##3
                self.get_features_index(Feature.allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),  ##4
                self.get_features_index(Feature.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),  ##5
                self.get_features_index(Feature.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),#6
                self.get_features_index(Feature.allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),#7
                Feature.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),  ##8
                self.get_features_index(Feature.allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),  # 9
                Feature.allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),##
                Feature.allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
                Feature.allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
                Feature.allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
                Feature.allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
                Feature.allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
                g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.  ##
            ])



        return atom_features_list,atom_coor_list  ##list  nparray

    def get_real_bonds(self,mol):
        real_bonds_begin=[]
        real_bonds_end=[]
        bond_features=[]

        for bond in mol.GetBonds():

        #     if int(bond.GetBeginAtom().GetAtomicNum()-1)==a[bond.GetBeginAtomIdx()][0] and int(bond.GetEndAtom().GetAtomicNum()-1)==a[bond.GetEndAtomIdx()][0]:
        #         print('right')
        #     else:
        #         print('wrong')

            real_bonds_begin.extend([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
            real_bonds_end.extend([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
            bond_features.append([
                self.get_features_index(Feature.bond_features['is_aromoatic'],bond.GetIsAromatic()),
                self.get_features_index(Feature.bond_features['is_conjugated'], bond.GetIsConjugated()),
                self.get_features_index(Feature.bond_features['is_in_ring'], bond.IsInRing()),
                self.get_features_index(Feature.bond_features['type'],int(bond.GetBondTypeAsDouble()*2))
            ])
            bond_features.append([
                self.get_features_index(Feature.bond_features['is_aromoatic'], bond.GetIsAromatic()),
                self.get_features_index(Feature.bond_features['is_conjugated'], bond.GetIsConjugated()),
                self.get_features_index(Feature.bond_features['is_in_ring'], bond.IsInRing()),
                self.get_features_index(Feature.bond_features['type'],int(bond.GetBondTypeAsDouble()*2))
            ])
        ##

        return real_bonds_begin,real_bonds_end,bond_features ##




if __name__=='__main__':
    print('***********************test*************************')

    datapath='./1fm9'
    data1=datapath+'/1fm9_pocket_reduce.pdb'
    data2=datapath+'/1fm9_pocket.pdb'
    data3=datapath+'/1fm9_ligand.sdf'


    data4='./1cgl/1cgl_ligand.sdf'
    data5='./11gs/11gs_pocket.pdb'
    data6='./11gs/11gs_ligand.mol'

    class_features=Feature()
    pocket = Chem.MolFromPDBFile(data5)
    ligand = Chem.MolFromMolFile(data6)##
    ##sanitize=False, removeHs=False
    a,b=class_features.get_features(pocket)
    begin,end,b_f=class_features.get_real_bonds(pocket)
    ##ligand = Chem.SDMolSupplier('%s/%s/%s_ligand.sdf' % (path, name, name), sanitize=False, removeHs=False)[0]
    for i in range(len(a)):
        if a[i][8]==1:
            print(i)


    print('***********************test*************************')

    # print(bond.GetIdx(), end='\t')
    # print(bond.GetBondType(), end='\t')
    # print(bond.GetBondTypeAsDouble(), end='\t')
    # print(bond.GetIsAromatic(), end='\t')
    # print(bond.GetIsConjugated(), end='\t')
    # print(bond.IsInRing(), end='\t')
    # print(bond.GetBeginAtomIdx(), end='\t')
    # print(bond.GetEndAtomIdx())





