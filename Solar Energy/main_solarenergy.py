import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.mat_gen import uniform
import reservoirpy as rpy
from functools import reduce

np.random.seed(2024)
rpy.set_seed(2341)

system = 'solar'
data_train = np.loadtxt(system+'_dataset_train.txt')
data_target = np.loadtxt(system+'_dataset_target.txt')

pre_steps = 10
target_dim = [64, 72, 103]
Standard = 'standard'
dimforpre_num = 11

umin = -0.18
umax = 0.18
init_mode = 'uniform'
connectivity = [0.2, 0.2, 0.2]
Win_init = uniform(low=umin, high=umax, connectivity=connectivity[0])
bias_init = uniform(low=umin, high=umax, connectivity=connectivity[1])
W_init = uniform(low=umin, high=umax, connectivity=connectivity[2])
Reservoir_size = 600
input_con, re_con = 0.2, 0.2
lr = 1
sr = 1
ridge = 1e-01

from nodes_expansion import Square_node
path_num = len(target_dim)*(pre_steps+1)
model_generate = 'MODEL='
for i in range(path_num):
    exec(f"IR{i} = Reservoir(Reservoir_size, lr=lr, sr=sr, Win=Win_init, bias=bias_init, W=W_init, "
         f"input_connectivity=input_con, rc_connectivity=re_con, name='IR'+str(i))")
    exec(f"RO{i} = Ridge(ridge=ridge, name='RO'+str(i))")
    exec(f"square_node{i} = Square_node(name='square_node'+str(i))")
    exec(f'path{i} = IR{i} >> square_node{i} >> RO{i}')
    if i == 0:
        model_generate += 'path0'
    else:
        model_generate += '&path'+str(i)
exec(model_generate)

def slidwindowlabel_maker(data_train, target_dim, forpre_steps, extend=False):
    rn = data_train.shape[0]-forpre_steps
    lag_label_mat = np.zeros((rn, len(target_dim)*(forpre_steps+1)))
    col = 0
    for dim_ind in target_dim:
        targ = data_train[:, dim_ind]
        for lag in range(forpre_steps+1):
            lag_label_mat[:, col] = targ[lag:rn+lag]
            col += 1
    if extend == True:
        lag_label_mat = np.c_[lag_label_mat, lag_label_mat[:, 0:forpre_steps+1]]
    return lag_label_mat

def data_processer(data_train, data_target, presteps, targetdim):
    wholetraindata = data_train
    datafortrain = data_train[:-presteps, :]
    dataforlabel = slidwindowlabel_maker(data_train, targetdim, presteps, extend=False)
    targetdata = np.vstack([data_train[-1, targetdim], data_target])
    lis = []
    for col_ind in range(targetdata.shape[1]):
        lis.append(targetdata[:, col_ind])
    datafortarget = reduce(np.append, lis)
    return datafortrain, dataforlabel, datafortarget, wholetraindata

from sklearn.feature_selection import mutual_info_regression
def select_relevant_dim(col_ind, data, allsort=True, n=5):
    MI = mutual_info_regression(data, data[:, col_ind])
    MI = np.abs(MI)
    all_rele_ind = np.argsort(-MI)
    if allsort==True:
        return all_rele_ind
    else:
        most_rele_ind = all_rele_ind[:n]
        return most_rele_ind

from sklearn.preprocessing import MinMaxScaler, StandardScaler
def fit_and_predict(models, train_data, train_label, whole_data, MI_dict, targetdim, presteps,
                    standardization=None, dp_num=4, tunebyzerolag=1, calcu_err_before_restore=True):

    pathnum = train_label.shape[1]
    train_dict = {}
    label_dict = {}
    warm_dict = {}
    pred_dict = {}
    reledim_dict = {}

    endvarind = int(pathnum/(presteps+1))
    varlist = list(np.linspace(0, endvarind-1, endvarind).astype(int))
    varlist.append(0)

    if standardization != None:
        if standardization == 'standard':
            scaler = StandardScaler()
            if calcu_err_before_restore == False:
                whole_data_ori = whole_data.copy()  # for pred_benchmark
            whole_data = scaler.fit_transform(whole_data)
            subtractor = scaler.mean_
            divisor = np.sqrt(scaler.var_)
        if standardization == 'minmax':
            scaler = MinMaxScaler()
            if calcu_err_before_restore == False:
                whole_data_ori = whole_data.copy()
            whole_data = scaler.fit_transform(whole_data)
            subtractor = scaler.data_min_
            divisor = scaler.data_range_
        train_data = scaler.transform(train_data)

        subtractor_targdim = subtractor[targetdim]
        divisor_targdim = divisor[targetdim]
        for i in range(len(targetdim)):
            train_label[:, i*(presteps+1):(i + 1)*(presteps+1)] = (train_label[:, i*(presteps+1):(i + 1)*(presteps+1)]-subtractor_targdim[i])/divisor_targdim[i]
    train_label_ext = np.c_[train_label, train_label[:, 0:presteps + 1]]

    for path in range(pathnum):
        div = divmod(path, presteps+1)
        startvar_ind = div[0]
        endvar_ind = varlist[startvar_ind+1]
        startvar_lag = div[1]
        reledimnum_s, reledimnum_e = dp_num*(presteps+1-startvar_lag), dp_num*startvar_lag
        colind_s, colind_e = targetdim[startvar_ind], targetdim[endvar_ind]  # 起始终止变量对应的列
        reledim_s = MI_dict[colind_s][:reledimnum_s]
        reledim_e = MI_dict[colind_e][:reledimnum_e]
        rele_dim_ind = np.concatenate((reledim_s, reledim_e))
        reledim_dict[path] = rele_dim_ind

        pathtrain = train_data[:, rele_dim_ind]
        pathlabel = train_label_ext[:, path:path+presteps+1]
        train_dict['IR'+str(path)] = pathtrain
        label_dict['RO'+str(path)] = pathlabel

    models = models.fit(train_dict, label_dict, warmup=5)

    for path in range(pathnum):
        warm_dict['IR' + str(path)] = whole_data[:-presteps, reledim_dict[path]]
    models.run(warm_dict, reset=True)

    if standardization != None and calcu_err_before_restore == False:
        pred_benchmark = whole_data_ori[-presteps:, targetdim]
    else:
        pred_benchmark = whole_data[-presteps:, targetdim]
    prediction = np.zeros((presteps+1, len(targetdim)))
    aggregated_preds_num = [presteps]
    for s in range(presteps):
        for path in range(pathnum):
            pred_dict['IR' + str(path)] = whole_data[-(presteps-s), reledim_dict[path]]
        modelpre_dict = models.call(pred_dict)

        pred_mat = np.zeros((1, presteps + 1))
        for i in range(path_num):
            pred = modelpre_dict['RO' + str(i)]
            pred_mat = np.r_[pred_mat, pred]
        pred_mat = pred_mat[1:, :]

        pred_restruc = np.zeros_like(pred_mat)
        pred_mat_ext = np.r_[pred_mat[-presteps:, :], pred_mat]
        for i in range(pathnum):
            r = i
            for j in range(presteps + 1):
                pred_restruc[i, j] = pred_mat_ext[r, presteps - j]
                r += 1
        prediction_s = np.mean(pred_restruc, axis=1)
        if calcu_err_before_restore == False:
            for i in range(len(targetdim)):
                prediction_s[i*(presteps+1):(i+1)*(presteps+1)] = prediction_s[i*(presteps+1):(i+1)*(presteps+1)] * divisor_targdim[i] + subtractor_targdim[i]
        prediction_s = prediction_s.reshape((-1, presteps + 1)).T  # predictions.shape=(len(target_dim), presteps+1)
        err = pred_benchmark[s:, :] - prediction_s[:-(s+1), :]
        mean_err = np.mean(err, axis=0)
        prediction_s[-(s + 2):, :] += mean_err
        prediction[:(s + 2), :] += prediction_s[-(s + 2):, :]
        aggregated_preds_num.append(presteps - s)
    aggregated_results_num = np.array(aggregated_preds_num)
    aggregated_coef = np.diag(1 / aggregated_results_num)
    prediction = np.dot(aggregated_coef, prediction)
    prediction = prediction.ravel(order='F')

    if tunebyzerolag != 0:
        for i in range(len(targetdim)):
            targ_dim = targetdim[i]
            zerolag_targ = whole_data[-1, targ_dim]
            pre_ind = i * (presteps + 1)
            prediction[pre_ind:pre_ind + presteps + 1] += (zerolag_targ - prediction[pre_ind]) * tunebyzerolag

    if standardization != None and calcu_err_before_restore == True:
        for i in range(len(targetdim)):
            prediction[i*(presteps+1):(i+1)*(presteps+1)] = prediction[i*(presteps+1):(i+1)*(presteps+1)] * divisor_targdim[i] + subtractor_targdim[i]
    return prediction


def rmse(x, y):
    return np.sqrt(np.sum(np.square(x-y), axis=0)/x.shape[0])
def mae(x, y):
    return np.sum(np.abs(x-y), axis=0)/x.shape[0]
from scipy.stats import pearsonr

def eval_model(target, pred, varnums, presteps, evalzerolag=True, plot=True, predname='prediction'):
    targetlist, predlist = [], []
    for i in range(varnums):
        targetlist.append(target[i*(presteps+1):(i+1)*(presteps+1)])
        predlist.append(pred[i*(presteps+1):(i+1)*(presteps+1)])
    if evalzerolag == False:
        for i in range(varnums):
            targetlist[i] = targetlist[i][1:]
            predlist[i] = predlist[i][1:]
        target = reduce(np.append, targetlist)
        pred = reduce(np.append, predlist)
    eachrmse, eachmae = [], []
    eachpcc = []
    for i in range(varnums):
        eachrmse.append(rmse(targetlist[i], predlist[i]))
        eachmae.append(mae(targetlist[i], predlist[i]))
        eachpcc.append(pearsonr(targetlist[i], predlist[i])[0])
    RMSE = rmse(target, pred)
    print('RMSE of Each variable：{};  Overall RMSE：{}'.format(eachrmse, RMSE))
    print('MAE of Each variable：{}'.format(eachmae))
    print('PCC of Each variable：{}'.format(eachpcc))
    if plot == True:
        colors = ['tomato', 'orange', 'gold', 'springgreen', 'deepskyblue'][::-1]
        plt.figure(figsize=(8, 6))
        for i in range(varnums):
            l1, = plt.plot(targetlist[i], color=colors[i], linewidth=1.0, linestyle='-', marker='s')
            l2, = plt.plot(predlist[i], color=colors[i], linewidth=1.0, linestyle='--', marker='o')
            plt.legend(handles=[l1, l2], labels=['target', predname], loc='best')
        plt.show()
        for i in range(varnums):
            plt.figure(figsize=(8, 6))
            l1, = plt.plot(targetlist[i], color=colors[i], linewidth=1.0, linestyle='-', marker='s')
            l2, = plt.plot(predlist[i], color=colors[i], linewidth=1.0, linestyle='--', marker='o')
            plt.legend(handles=[l1, l2], labels=['target', predname], loc='best')
            plt.show()
    return eachrmse, RMSE, eachmae, eachpcc


def main(model, TRAIN, TARGET, targetdim, presteps, dimforpre_num=4, evalzerolag=False, standardization=None):
    train, label, target, wholetrain = data_processer(TRAIN, TARGET, presteps, targetdim)

    MIdict = {}
    for i in range(len(target_dim)):
        MIsorted = select_relevant_dim(targetdim[i], wholetrain)
        MIdict[targetdim[i]] = MIsorted

    preds = fit_and_predict(model, train, label, wholetrain, MIdict, targetdim, presteps, standardization=standardization, dp_num=dimforpre_num)
    eachrmse, RMSE, eachmae, eachpcc = eval_model(target, preds, len(target_dim), presteps, evalzerolag=evalzerolag)
    return preds, eachrmse, RMSE, eachmae, eachpcc

final_pred, eachRMSE, RMSE, eachMAE, eachPCC = main(MODEL, data_train, data_target, target_dim, pre_steps,
                                  dimforpre_num=dimforpre_num, evalzerolag=True, standardization=Standard)

del MODEL
for i in range(path_num):
    exec(f'del path{i}, IR{i}, square_node{i}, RO{i}')
