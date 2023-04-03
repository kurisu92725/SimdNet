import argparse
from models.simdnet import Simd
from models.graphdataset import collate_6,PDBbind_2016_complex
from simd_train import try_gpu
import pickle
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import  nn
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_(loader,net,loss_f):

    net.eval()
    # with torch.no_grad():
    l_test=0.0

    label=[]
    pre=[]

    for i, (g, y)  in enumerate(loader):
        with torch.no_grad():
            g, y = g.to(device), y.to(device)
            y_hat = net(g)
            l = loss_f(y_hat, y)
            label.extend(y.tolist())
            pre.extend(y_hat.tolist())

            l_test+=float(l)


    return label,pre,l_test/(i+1)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=int, default=0, help="GPU id for training model, else use CPU")

    argparser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    
    #May need to change files
    argparser.add_argument('--test_path', type=str, default='./dataset/core',help='path of testing set ')
    argparser.add_argument('--test_name_path', type=str, default='./dataset/core_name_pointer.pkl', help='path of testing name')
    argparser.add_argument('--label_path', type=str, default='./pdb_2016_labels.pkl',help='label path')

    argparser.add_argument('--param_path', type=str, default='./example_model/net_params_0', help='param path')  ##need change
    argparser.add_argument('--saving_path', type=str, default='./result_dict', help='saving path')##need change


    args = argparser.parse_args()
    #params
    gpuid,batch_size=args.gpuid,args.batch_size
    test_path,test_name,label_files=args.test_path,args.test_name_path,args.label_path
    save_files = args.saving_path
    params_path = args.param_path


    ##dataset
    test_dataset = PDBbind_2016_complex(test_name, test_path, label_files)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_6)


    ##net


    device = try_gpu(gpuid)
    print(device)
    net = Simd()
    net.load_state_dict(torch.load(params_path))
    net.to(device)

    loss_f0 = nn.MSELoss()

    result_file = save_files+'/simd_core_2016_result.pkl'

    save_ = {}

    label, pre, avg_loss = test_(test_loader,net,loss_f0)

    x = np.array(label)
    y = np.array(pre)

    pccs = pearsonr(x, y)
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    mape = mean_absolute_percentage_error(x, y)

    save_['label'] = label
    save_['pre'] = pre
    save_['pccs'] = pccs
    save_['rmse'] = rmse
    save_['mae'] = mae
    save_['mape'] = mape

    tmp_file = open(result_file, 'wb')
    pickle.dump(save_, tmp_file)
    tmp_file.close()

    print(f'rmse:{rmse} mae:{mae}  mape:{mape}  pcss:{pccs}  || avg_loss:{avg_loss}')


    ###plot
    data = np.stack((save_['label'], save_['pre']), axis=0)
    # label=np.array(tmp_dict['label'])
    # affinity_pre=np.array(tmp_dict['pre'])
    # data=sns.load_dataset(data)
    c1 = sns.xkcd_rgb['blue']
    c2 = sns.xkcd_rgb['salmon']
    c3 = sns.xkcd_rgb['orchid']
    df = pd.DataFrame({'Prediction': save_['pre'], 'True': save_['label']})
    sns.set(style="darkgrid")
    ax = sns.jointplot(data=df, x='True', y='Prediction', kind='reg', color=c3, xlim=(0, 14), ylim=(0, 14))
    plt.text(1.4, 13, 'Core Set v2016', fontsize=20)
    plt.text(1.4, 12, r'$R_{p}=$'+str(pccs[0]), fontsize=20)
    # plt.legend(loc='best')

    plt.show()


    print('now finished!')

