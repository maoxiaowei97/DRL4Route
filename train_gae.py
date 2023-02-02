# -*- coding: utf-8 -*-
import os
import argparse
from my_utils.eval import *
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from tqdm import  tqdm
import torch
from my_utils.utils_gae import  to_device, get_nonzeros, dict_merge
from algorithm.logistics_pickup.deeproute_a2c.Dataset import DeepRouteDataset

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from my_utils.eval import Metric
    model.eval()

    evaluator_1 = Metric([1, 5])
    evaluator_2 = Metric([1, 11])
    evaluator_3 = Metric([1, 15])
    evaluator_4 = Metric([1, 25])
    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            E_static_fea, V, V_reach_mask, V_dispatch_mask, \
            E_mask, label, label_len, V_len, start_fea, start_idx = batch
            outputs, pointers, decoder_outputs_T, test_states, scores = model(V, V_reach_mask, V_len, sample = False, type = 'mle')

            pred_steps, label_steps, labels_len, preds_len = \
                get_nonzeros(pointers.reshape(-1, outputs.size()[-1]), label.reshape(-1, outputs.size()[-1]),
                             label_len.reshape(-1), V_len.reshape(-1), pad_value)

            evaluator_1.update(pred_steps, label_steps, labels_len, preds_len)
            evaluator_2.update(pred_steps, label_steps, labels_len, preds_len)
            evaluator_3.update(pred_steps, label_steps, labels_len, preds_len)
            evaluator_4.update(pred_steps, label_steps, labels_len, preds_len)

    if mode == 'val':
        return evaluator_4

    params_1 = dict_merge([evaluator_1.to_dict(), params])
    params_1['eval_min'] = 1
    params_1['eval_max'] = 5
    save2file(params_1)

    print(evaluator_2.to_str())
    params_2 = dict_merge([evaluator_2.to_dict(), params])
    params_2['eval_min'] = 1
    params_2['eval_max'] = 11
    save2file(params_2)

    print(evaluator_3.to_str())
    params_3 = dict_merge([evaluator_3.to_dict(), params])
    params_3['eval_min'] = 1
    params_3['eval_max'] = 15
    save2file(params_3)

    print(evaluator_4.to_str())
    params_4 = dict_merge([evaluator_4.to_dict(), params])
    params_4['eval_min'] = 1
    params_4['eval_max'] = 25
    save2file(params_4)

    return evaluator_4

def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    E_static_fea, V, V_reach_mask, V_dispatch_mask, \
    E_mask, label, label_len, V_len, start_fea, start_idx = batch

    pred_scores, pred_pointers, decoder_outputs_T, test_states, greedy_values = model(V, V_reach_mask, V_len, sample=False, type='mle')
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    N = pred_pointers.size(-1)
    pad_value = N - 1
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=pad_value)
    rl_log_probs, sample_out, values = model(V, V_reach_mask, V_len, sample=True, type='rl')
    with torch.autograd.no_grad():
        _, greedy_out, _ = model(V, V_reach_mask, V_len, sample=False, type='rl')

    loss = mle_loss

    return pred_pointers, pred_scores, loss, sample_out, greedy_out, label, V_reach_mask, V_len, rl_log_probs, values, test_states

def main(params):

    params['model'] = 'deeproute_gae_logistics'
    params['hs'] = 16
    params['sort_x_size'] = 8
    params['pad_value'] = params['max_task_num'] - 1
    params['dqn_pretrain'] = False
    params['a2c_train'] = True
    from my_utils.utils_gae import RL
    rl = RL()
    rl.run(params, DeepRouteDataset, process_batch, test_model)

def get_params():
    from my_utils.utils_gae import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='deeproute_gae_logistics')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--sort_x_size', type=int, default=8) #number of features in the node
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    import time, nni
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
