import argparse
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import  nn
from torch.nn import functional as F
import os
from models.simdnet import Simd
from models.simdnet import swish
from models.graphdataset import dataset_for_pyg,collate_6
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



actdic={'swish':swish,'relu':F.relu,'leakyrelu':F.leaky_relu,'sigmoid':torch.sigmoid,'tanh':torch.tanh}

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_1(net,train_loader,optimizer,loss_f,epoch):
    net.train()

    loss_1=0.0
    for i, (g, y) in enumerate(train_loader):
        optimizer.zero_grad()


        g, y = g.to(device), y.to(device)
        y_hat = net(g)
        l1 = loss_f(y_hat, y)

        l1.backward()
        loss_1+=float(l1)

        optimizer.step()
        #The output of the following line of code will only appear in the initial epoch.
        if epoch==0:
            print(f'\n now {i} batch')


    return loss_1/(i+1)


def test_1(net,test_loader,loss_f):

    net.eval()
    # with torch.no_grad():
    l_test=0.0

    for i, (g, y)  in enumerate(test_loader):
        with torch.no_grad():
            g, y = g.to(device), y.to(device)
            y_hat = net(g)
            l = loss_f(y_hat, y).float()
            # print("loss: ",float(l))
            l_test+=float(l)


    return l_test/(i+1)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=int, default=0, help="GPU id for training model, else use CPU")
    argparser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")##5e-4
    argparser.add_argument('--epochs', type=int, default=1000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    argparser.add_argument('--optimizer', type=int, default=0, help="Adam if 0, AdamW if 1")

    argparser.add_argument('--num_layers', type=int, default=4, help="Layers number")
    argparser.add_argument('--cutoff', type=float, default=5.0, help="Cutoff distance for atomic interactions when building graph, also as a parameter of the network")
    argparser.add_argument('--hidden_channels', type=int, default=128, help="Hidden embedding size")
    argparser.add_argument('--out_emb_channels', type=int, default=256, help="Embedding size used in mid layers in NodeUpdate block")
    argparser.add_argument('--out_channels', type=int, default=128,help="Output embedding size of each atom after each NodeUpdate block")
    argparser.add_argument('--num_output_layers', type=int, default=3,help="Number of mid layers in each NodeUpdate block")

    argparser.add_argument('--num_spherical', type=int, default=7,help="Number of spherical harmonics.")
    argparser.add_argument('--num_radial', type=int, default=6,help="Number of radial basis functions")
    argparser.add_argument('--envelope_exponent', type=int, default=5,help="Shape of the smooth cutoff, p-1")


    argparser.add_argument('--basis_emb_size', type=int, default=8, help="Embedding size used in the basis transformation")
    argparser.add_argument('--int_emb_size', type=int, default=64, help="Embedding size used for interaction triplets")
    argparser.add_argument('--res_number', type=int, default=3, help="Number of Residual Layers")
    argparser.add_argument('--act_function', type=str, default='swish', help="swish,relu,leakyrelu,sigmodi, or tanh")

    argparser.add_argument('--readout_inputsize', type=int, default=512, help="Readout input size, four times out_channels")
    argparser.add_argument('--outdim1', type=int, default=64, help="Embedding size of the mid layer when reducting dimension")
    argparser.add_argument('--output_init', type=str, default='GlorotOrthogonal',help="Output init")

    #The file in which the preprocessed dataset resides
    argparser.add_argument('--train_valid_path', type=str, default='../dataset_pre/data11/refine',help='path of training set ')
    argparser.add_argument('--train_valid_name_path', type=str, default='../dataset_pre/data11/refine_name_pointer.pkl',help='path of training name')
    argparser.add_argument('--test_path', type=str, default='../dataset_pre/data11/core',help='path of testing set ')
    argparser.add_argument('--test_name_path', type=str, default='../dataset_pre/data11/core_name_pointer.pkl', help='path of testing name')
    argparser.add_argument('--label_path', type=str, default='./pdb_2016_labels.pkl',help='label path')

    argparser.add_argument('--saving_path', type=str, default='./netpar_result/simd_', help='saving path')


    args = argparser.parse_args()
    #params
    gpuid,lr,epochs,batch_size=args.gpuid,args.lr,args.epochs,args.batch_size
    train_valid_path,train_valid_name,test_path,test_name,label_files=args.train_valid_path,args.train_valid_name_path,args.test_path,args.test_name_path,args.label_path
    save_files = args.saving_path

    #net params
    cutoff = args.cutoff
    num_layers = args.num_layers
    hidden_channels = args.hidden_channels
    out_channels = args.out_channels
    int_emb_size = args.int_emb_size
    basis_emb_size = args.basis_emb_size
    out_emb_channels = args.out_emb_channels
    num_spherical = args.num_spherical
    num_radial = args.num_radial
    envelope_exponent = args.envelope_exponent
    num_before_skip = 1
    num_after_skip = args.res_number-num_before_skip
    num_output_layers = args.num_output_layers
    output_init = args.output_init
    act = actdic[args.act_function]


    #dataset
    train_set, valid_set, test_set = dataset_for_pyg(train_valid_name, train_valid_path, test_name, test_path,label_files)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_6)
    #The results of the validation set can be used to control the learning
    # rate and the stopping time of network training, but are not used because the molecular space of the molecule or complex is too large.
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_6)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_6)
    index_dic = {}
    index_dic['train'] = train_loader.dataset.indices
    index_dic['valid'] = valid_loader.dataset.indices


    #net
    net=Simd(cutoff=cutoff,num_layers=num_layers,hidden_channels=hidden_channels, out_channels=out_channels,
             int_emb_size=int_emb_size, basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels,
             num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent,
             num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_output_layers=num_output_layers,act=act, output_init=output_init,)

    device = try_gpu(gpuid)

    print(device)

    net.to(device)

    if args.optimizer ==0:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    loss_f0 = nn.MSELoss()

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_test_loss = []
    tmp_loss = []

    best_val_ls = None
    best_test_ls = None
    best_net_stat = None
    loss_dict = {}


    os.makedirs(save_files)
    tmp_file = open(save_files + '/split_list.pkl', 'wb')
    pickle.dump(index_dic, tmp_file)
    tmp_file.close()

    for epoch in tqdm(range(epochs)):
        train_l = train_1(net,train_loader,optimizer,loss_f0,epoch)
        v_l = test_1(net,valid_loader,loss_f0)
        test_l = test_1(net,test_loader,loss_f0)

        #If you don't need to analyze or save network parameters, comment out the next line of code.
        torch.save(net.state_dict(), save_files + '/net_params_%d' % (epoch))



        if best_val_ls is None or v_l <= best_val_ls:
            best_val_ls = v_l
        if best_test_ls is None or test_l <= best_test_ls:
            best_test_ls = test_l
        epoch_train_loss.append(train_l)
        epoch_val_loss.append(v_l)
        epoch_test_loss.append(test_l)
        tmp_loss.append([train_l, v_l, test_l])
        tmp_file = open(save_files + '/tmp_result.pkl', 'wb')
        pickle.dump(tmp_loss, tmp_file)
        tmp_file.close()

        print(f'Epoch: {epoch:03d},  Train Loss: {train_l:.7f},   '
              f'Val MSE: {v_l:.7f}, Test MSE: {test_l:.7f} , '
              f'Best Val MSE: {best_val_ls:.7f}, Best Test MSE: {best_test_ls:.7f}')

    loss_dict['train_loss'] = epoch_train_loss
    loss_dict['valid_loss'] = epoch_val_loss
    loss_dict['test_loss'] = epoch_test_loss
    loss_dict['best_val_loss'] = best_val_ls
    loss_dict['best_test_loss'] = best_test_ls
    torch.save(net.state_dict(), save_files + '/net_params_final')

    result = open(save_files + '/result.pkl', 'wb')
    pickle.dump(loss_dict, result)
    result.close()
    print('all task finished!')




