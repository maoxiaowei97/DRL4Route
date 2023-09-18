# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from my_utils.utils import *
from algorithm.DRL4Route.Dataset import DRL4RouteDataset

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from my_utils.eval import Metric
    model.eval()

    evaluator_1 = Metric([params['eval_start'], params['eval_end_1']])
    evaluator_2 = Metric([params['eval_start'], params['eval_end_2']])

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, label, label_len = batch
            outputs, pointers, _ = model(V, V_reach_mask, sample = False, type = 'mle')

            pred_steps, label_steps, labels_len = get_samples(pointers.reshape(-1, outputs.size()[-1]), label.reshape(-1, outputs.size()[-1]),
                             label_len.reshape(-1), pad_value)

            evaluator_1.update(pred_steps, label_steps, labels_len)
            evaluator_2.update(pred_steps, label_steps, labels_len)

    if mode == 'val':
        return evaluator_2

    params_1 = dict_merge([evaluator_1.to_dict(), params])
    params_1['eval_min'] = params['eval_start']
    params_1['eval_max'] = params['eval_end_1']
    save2file(params_1)

    print(evaluator_2.to_str())
    params_2 = dict_merge([evaluator_2.to_dict(), params])
    params_2['eval_min'] = params['eval_start']
    params_2['eval_max'] = params['eval_end_2']
    save2file(params_2)


    return evaluator_2

def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    V, V_reach_mask, label, label_len = batch

    pred_scores, pred_pointers, values = model(V, V_reach_mask, sample=False, type='mle')
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    N = pred_pointers.size(-1)
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
    rl_log_probs, sample_out, sample_values = model(V, V_reach_mask, sample=True, type='rl')
    with torch.autograd.no_grad():
        _, greedy_out, _ = model(V, V_reach_mask, sample=False, type='rl')
    if params['model'] == 'DRL4Route_REINFORCE':
        seq_pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

        sample_out_samples, greedy_out_samples, label_samples, label_len_samples, rl_log_probs_samples, seq_pred_len_samples = \
            get_reinforce_samples(sample_out.reshape(-1, N), greedy_out.reshape(-1, N), label.reshape(-1, N), label_len.reshape(-1), params['pad_value'], rl_log_probs, seq_pred_len)

        log_prob_mask = get_log_prob_mask(seq_pred_len_samples, params)

        rl_log_probs_samples = rl_log_probs_samples * log_prob_mask

        rl_log_probs_samples = torch.sum(rl_log_probs_samples, dim=1) / seq_pred_len_samples

        krc_reward, lsd_reward, acc_3_reward = calc_reinforce_rewards(sample_out_samples, label_samples, label_len_samples, params)

        baseline_krc_reward, baseline_lsd_reward, baseline_acc_3_reward = calc_reinforce_rewards(greedy_out_samples, label_samples, label_len_samples, params)

        reinforce_loss = -torch.mean(torch.tensor(baseline_lsd_reward - lsd_reward).to(rl_log_probs_samples.device) * rl_log_probs_samples)

        loss = mle_loss + params['rl_ratio'] * reinforce_loss
    else:
        loss = mle_loss

    return pred_pointers, pred_scores, loss, sample_out, greedy_out, label, V_reach_mask, rl_log_probs, sample_values

def main(params):
    trainer = DRL4Route()
    trainer.run(params, DRL4RouteDataset, process_batch, test_model)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args
