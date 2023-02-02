# -*- coding: utf-8 -*-
import numpy as np
import os
import math
import torch.nn as nn
from tqdm import  tqdm
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import nni, time
import algorithm.logistics_pickup.deeproute_a2c.critic as critic
import algorithm.logistics_pickup.deeproute_a2c.ActorCritic as ac
import traceback
os.environ['MKL_SERVICE_FORCE_INTEL']='1'
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,4,5,6,7'
def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def whether_stop(metric_lst = [], n=2, mode='maximize'):
    '''
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    '''
    if len(metric_lst) < 1:return False # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx,v in enumerate(metric_lst):
        if v == max_v:max_idx = idx
    return max_idx < len(metric_lst) - n

from multiprocessing import Pool
def multi_thread_work(parameter_queue,function_name,thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return  result

class EarlyStop():
    """
    For training process, early stop strategy
    """
    def __init__(self, mode='maximize', patience = 1):
        self.mode = mode
        self.patience =  patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1 # the best epoch
        self.is_best_change = False # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        #update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        #update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize'  else self.metric_lst.index(min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch#update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:return -1
        else:
            return self.metric_lst[self.best_epoch]
def get_len_for_or_tools(init_mask_i, dis_i):
        msk = init_mask_i.clone()
        j, point = 0, 0
        while not msk.all():
            dis_j = dis_i[point].masked_fill(msk, 1e6)
            idx = torch.argmin(dis_j)
            if idx % 2 != 0:
                msk[idx + 1] = 0
            msk[idx], point = 1, idx
            j += 1
        return j


def batch_file_name(file_dir, suffix='.train'):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L

# merge all the dict in the list
def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def get_dataset_path(params = {}):
    dataset = params['dataset']
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

def write_list_list(fp, list_, model="a", sep=","):
    dir = os.path.dirname(fp)
    if  not os.path.exists(dir): os.makedirs(dir)
    f = open(fp,mode=model,encoding="utf-8")
    count=0
    lines=[]
    for line in list_:
        a_line=""
        for l in line:
            l=str(l)
            a_line=a_line+l+sep
        a_line = a_line.rstrip(sep)
        lines.append(a_line+"\n")
        count=count+1
        if count==10000:
            f.writelines(lines)
            count=0
            lines=[]
    f.writelines(lines)
    f.close()

def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t
    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()
    # write_to_hdfs(file_name, head)
    # 写数据
    with open(file_name, "a", newline='\n') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        # params['log_time'] = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) #不同服务器上时间不一样
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)
        # write_to_hdfs(file_name, data)


#----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader
def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')

    # dataset
    parser.add_argument('--min_task_num', type=int, default=0, help = 'minimal number of task')
    parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task')
    parser.add_argument('--dataset', default='logistics_0831', type=str, help='food_cou or logistics')#logistics_0831, logistics_decode_mask
    parser.add_argument('--pad_value', type=int, default=24, help='logistics: max_num - 1, pd: max_num + 1')
    parser.add_argument('--num_worker_pd', type=int, default=1000, help='number of workers in food delivery dataset')
    parser.add_argument('--num_worker_logistics', type=int, default=2346, help='number of workers in logistics dataset')

    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=9, help='early stop at')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 4)')
    parser.add_argument('--task', type=str, default='logistics', help='food_cou or logistics')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    parser.add_argument('--model_path', type=str, default=None, help='best model path in logistics')

    #common settings for graph2route model
    parser.add_argument('--node_fea_dim', type=int, default=8, help = 'dimension of node input feature')
    parser.add_argument('--edge_fea_dim', type=int, default=4, help = 'dimension of edge input feature')
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--gcn_num_layers', type=int, default=2)
    parser.add_argument('--k_nearest_neighbors', type=str, default='n')
    parser.add_argument('--k_min_nodes', type=int, default=3)
    parser.add_argument('--b', type=int, default=2)
    #for deeproute
    parser.add_argument('--sort_x_size', type=int, default=8)

    # settings for evaluation
    parser.add_argument('--eval_start', type=int, default=1)
    parser.add_argument('--eval_start_pd', type=int, default=3)
    parser.add_argument('--eval_end_1', type=int, default=11)
    parser.add_argument('--eval_end_2', type=int, default=25)

    return parser

def filter_data(data_dict={}, len_key = 'node_len',  min_len=0, max_len=20):
    '''
    filter data, For dataset
    '''
    new_dic = {}

    keep_idx = [idx for idx, l in enumerate(data_dict[len_key]) if l >= min_len and l <= max_len]
    for k, v in data_dict.items():
        new_dic[k] = [data for idx, data in enumerate(data_dict[k]) if idx in keep_idx]
    return new_dic

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch


def get_nonzeros(pred_steps, label_steps, label_len, pred_len, pad_value):
    pred = []
    label = []
    label_len_list = []
    pred_len_list = []
    for i in range(pred_steps.size()[0]):
        #label 不为0时才会考虑该测试该step
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].detach().cpu().numpy().tolist())
            pred.append(pred_steps[i].detach().cpu().numpy().tolist())
            label_len_list.append(label_len[i].detach().cpu().numpy().tolist())
            pred_len_list.append(pred_len[i].detach().cpu().numpy().tolist())
    return torch.LongTensor(pred), torch.LongTensor(label),\
           torch.LongTensor(label_len_list), torch.LongTensor(pred_len_list)

def get_model_function(model):
    model_dict = {}
    import algorithm.logistics_pickup.deeproute_ac.DeepRoute as deeproute_logistics_ac
    import algorithm.logistics_pickup.deeproute_a2c.DeepRoute as deeproute_logistics_a2c
    model_dict['deeproute_logistics_ac'] = (deeproute_logistics_ac.DeepRoute, deeproute_logistics_ac.save2file)
    model_dict['deeproute_logistics_a2c'] = (deeproute_logistics_a2c.DeepRoute, deeproute_logistics_a2c.save2file)
    model_dict['deeproute_gae_logistics'] = (deeproute_logistics_a2c.DeepRoute, deeproute_logistics_a2c.save2file)

    model, save2file = model_dict[model]
    return model, save2file

import math
def calc_reward(sample, label): #解码到padding值时如何处理
    #计算每一步的reward
    sample = sample.detach().cpu().tolist()
    label = label.detach().cpu().tolist()
    #1.遍历pred中的值 看pred中的值是否存在于label，若在，算下顺序差；若不在，看这个值在pred中的顺序，加上当前pred顺序和总长度的差
    reward = 20 #设定初始reward - 第t步动作的距离差
    valid_label_length = len(label)
    if valid_label_length == 0:
        print('error')
        return 0 #之后不会去掉
    else:
        idx_diff_list = []
        # for v in sample[-1]:
        if sample[-1] not in label: #只看最后一个动作
            if sample.index(sample[-1]) > valid_label_length - 1: #已经输出所有label, 其余pred随意输出
                # label: 0, 1, 2
                # pred: 2, 1, 3, 4, 5
                return 0
            else:
                idx_diff_list.append(math.fabs(valid_label_length - sample.index(sample[-1])))#valid_label_length - sample.index(v)
        else:
            idx_diff_list.append(math.fabs(label.index(sample[-1]) - sample.index(sample[-1])))
        idx_diff_list = list(map(lambda x: x ** 2, idx_diff_list))
        reward = reward - sum(idx_diff_list) / len(idx_diff_list)
        return reward

def calc_single_reward(sample, label):
    reward = []
    sample = sample[(sample != 24).nonzero().squeeze(1)]
    label = label[(label != 24).nonzero().squeeze(1)]
    for t in range(len(sample)): #for decode time steps
        reward.append(calc_reward(sample[:t + 1], label))
    return reward

class RL(object):
    def filter_non_label_sample(self, state, V_reach_mask, rl_log_prob, greedy_out, label, scores):
        V_reach_mask_list = []
        label_list = []
        rl_log_prob_list = []
        greedy_out_list = []
        state_list = []
        scores_list = []
        for i in range(V_reach_mask.shape[0]):
            # label 不为0时才会考虑该测试该step
            if label[i].min().item() != 24:
                label_list.append(label[i])
                V_reach_mask_list.append(V_reach_mask[i])
                rl_log_prob_list.append(rl_log_prob[i])
                greedy_out_list.append(greedy_out[i])
                state_list.append(state[i])
                scores_list.append(scores[i])
        return torch.stack(state_list), torch.stack(V_reach_mask_list), torch.stack(rl_log_prob_list), torch.stack(greedy_out_list), torch.stack(label_list), torch.stack(scores_list)

    def collect_update_modify(self, state, V_reach_mask, rl_log_prob, sample_out, label, scores, params):
        state, V_reach_mask, rl_log_prob, sample_out, label, scores = self.filter_non_label_sample(state, V_reach_mask.reshape(-1, 25), rl_log_prob, sample_out, label.reshape(-1, 25), scores) #样本所有step都没有label
        r_steps = []
        states = []
        state_value_list_1 = []
        rl_log_probs = []

        batch = sample_out.shape[0]
        label = label.reshape(batch, -1)
        for n in range(len(state)):
            r_steps.extend(calc_single_reward(sample_out[n], label[n]))  # reward is calculated over the best action through greedy search
        #r_steps只算了pred不为24的值对应的reward
        valid_index = (sample_out.reshape(-1) != 24).nonzero().squeeze(1).detach().cpu().tolist()
        done_index = []
        for k in range(len(label.reshape(-1, 25))):
            if k == 0:
                done_index.append((~V_reach_mask.reshape(-1, 25)[k] + 0).sum().item())
            else:
                done_index.append((~V_reach_mask.reshape(-1, 25)[k] + 0).sum().item() + done_index[-1])
        for i in range(len(r_steps)):
            states.append(state.reshape(-1, state.shape[-1])[valid_index[i]]) #在768 * 25个时间步中有效的index
            rl_log_probs.append(rl_log_prob.reshape(-1, 1)[valid_index[i]])  # 分成各个时间步的样本
            state_value_list_1.append(scores.reshape(-1, 1)[valid_index[i]])

        done_index.insert(0, 0) #加入起始点索引
        rewards_ = []
        advantages = []
        for sample_index in range(len(done_index) - 1):
            # sample_index = 0 #记录当前样本索引
            R = 0
            advantage = 0
            next_value = 0
            r_samples = r_steps[done_index[sample_index]: done_index[sample_index + 1]]
            v_samples = state_value_list_1[done_index[sample_index]: done_index[sample_index + 1]]
            rewards_sample = []
            advantage_sample = []
            for r, v in zip(r_samples[::-1], v_samples[::-1]):
            # for r in r_samples[::-1]:
                R = r + 0.99 * R
                if r == 0:
                    v = 0
                else:
                    v = v.item()
                rewards_sample.insert(0, R)
                td_error = r + next_value * 0.99 - v
                advantage = td_error + advantage * 0.99 * params['trace_decay']
                next_value = v
                advantage_sample.insert(0, advantage)
            advantages.extend(advantage_sample)
            rewards_.extend(rewards_sample)
        #去掉不评价step
        states_list = []
        rl_log_probs_list = []
        rewards_list = []
        scores_list_2 = []
        advantages_list_2 = []
        valid_index_2 = (torch.tensor(rewards_) != 0).nonzero().squeeze(1).tolist() # 得到评价steps的reward
        for i in range(len(valid_index_2)):
            states_list.append(states[valid_index_2[i]])
            rewards_list.append(rewards_[valid_index_2[i]])
            rl_log_probs_list.append(rl_log_probs[valid_index_2[i]])
            scores_list_2.append(state_value_list_1[valid_index_2[i]])
            advantages_list_2.append(advantages[valid_index_2[i]])

        returns = torch.tensor(rewards_list).to(self.device)
        returns = (returns - returns.mean()) / returns.std()

        advantages = torch.tensor(advantages_list_2).to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std()

        return torch.stack(states_list, dim=0), torch.stack(rl_log_probs_list, dim=0), returns, torch.stack(scores_list_2, dim=0), advantages, rewards_list


    def calc_value_loss(self, state_value, r_steps):
        value_losses = F.smooth_l1_loss(state_value, r_steps.detach()).sum()
        return value_losses

    def calc_policy_loss(self, rl_log_probs, r_steps, state_value, advantages):
        # td_error = r_steps - state_value
        policy_losses = (-rl_log_probs.squeeze(1) * advantages.detach()).mean()
        # policy_losses = -rl_log_probs.squeeze(1) * r_steps
        return policy_losses

    def run(self, params, DATASET, process_batch, test_model):
        self.time = time.time()

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        params['device'] = self.device
        self.critic_net = critic.Critic(params)
        self.critic_net.to(self.device)

        params['pad_value'] = params['max_task_num'] - 1
        params['train_path'], params['val_path'],  params['test_path'] = get_dataset_path(params)
        pprint(params)  # print the parameters

        train_dataset = DATASET(mode='train', params=params)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=None)  # num_workers=2,

        val_dataset = DATASET(mode='val', params=params)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=None)  # cfg.batch_size

        test_dataset = DATASET(mode='test', params=params)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=None)#, collate_fn=collate_fn

        model, save2file = get_model_function(params['model'])
        # model_path = '/home/zsxz/graph2route_release/algorithm/logistics_pickup/deeproute_ac/actor_models/actor_1673395209.5034876.params'
        model_path = '/home/zsxz/graph2route_release/data/dataset/logistics/sort_model/hidden_size-16.deeproute_scst_logistics1670220248.5822163.csv1670220248.582231'
        model = model(params)
        AC_model = ac.DeepRoute(params)
        AC_model_dict = AC_model.state_dict()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        pretrained_dict = torch.load(model_path, map_location=self.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in AC_model_dict}
        AC_model_dict.update(pretrained_dict)
        AC_model.load_state_dict(AC_model_dict)
        AC_model.to(self.device)

        optimizer = Adam(AC_model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
        save_model_path = f'/home/zsxz/graph2route_release/algorithm/logistics_pickup/deeproute_a2c/gae/gae_{self.time}.params'
        total_reward = []
        for epoch in range(params['num_epoch']):
            epoch_reward = []
            if early_stop.stop_flag: break
            postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
            with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
                for i, batch in enumerate(t):
                    ave_loss = None
                    ave_value_loss = None
                    if params['a2c_train'] == True:
                        pass
                        AC_model.train()
                        pred, pred_scores, mle_loss, sample_out, greedy_out, label, V_reach_mask, V_len, rl_log_probs, state_values, test_states = process_batch(batch, AC_model, self.device, params) #1.collect dqn transition
                        test_states, rl_log_probs, r_steps, state_values, advantages, b_reward_to_go = self.collect_update_modify(test_states, V_reach_mask, rl_log_probs, sample_out, label, state_values, params)
                        value_loss = self.calc_value_loss(state_values, r_steps)
                        policy_loss = self.calc_policy_loss(rl_log_probs, r_steps, state_values, advantages)
                        epoch_reward.extend(b_reward_to_go)

                        loss = params['rl_ratio'] * policy_loss + mle_loss * (1 - params['rl_ratio']) + value_loss * 0.1
                        # loss = params['rl_ratio'] * policy_loss + mle_loss * (1 - params['rl_ratio']) + value_loss * 0.05

                        if ave_loss is None:
                            ave_loss = loss.item()
                            ave_value_loss = value_loss.item()
                        else:
                            ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)
                            ave_value_loss = ave_value_loss * i / (i + 1) + value_loss.item() / (i + 1)
                        postfix["loss"] = ave_loss
                        postfix["current_loss"] = loss.item()
                        postfix["value_loss"] = value_loss
                        postfix["current_value_loss"] = value_loss.item()
                        t.set_postfix(**postfix)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total_reward.append(np.array(epoch_reward).mean())
                val_result = test_model(AC_model, val_loader, self.device, params['pad_value'], params, save2file, 'val')  # 对于验证集上，不需要写结果；
                print('\nval result:', val_result.to_str(), 'Best krc:', round(early_stop.best_metric(), 3),
                      '| Best epoch:', early_stop.best_epoch)
                is_best_change = early_stop.append(val_result.to_dict()['krc'])
                if is_best_change:
                    print('value:', val_result.to_dict()['krc'], early_stop.best_metric())
                    torch.save(AC_model.state_dict(), save_model_path)
                    print('best model saved')
                    print('model path:', save_model_path)
        try:
            print('loaded model path:', save_model_path)
            AC_model.load_state_dict(torch.load(save_model_path))
            print('best model loaded !!!')
        except:
            print('load best model failed')
        test_result = test_model(AC_model, test_loader, self.device, params['pad_value'], params, save2file, 'test')
        np.save(f'/home/zsxz/graph2route_release/algorithm/logistics_pickup/deeproute_a2c/gae/reward_0131/reward_{self.time}.npy', np.array(total_reward))
        print('\n-------------------------------------------------------------')
        print('Best epoch: ', early_stop.best_epoch)
        print(f'{params["model"]} Evaluation in test:', test_result.to_str())


if __name__ == '__main__':
    pass


