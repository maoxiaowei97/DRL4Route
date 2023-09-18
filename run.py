# -*- coding: utf-8 -*-
import os
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
os.environ['MKL_SERVICE_FORCE_INTEL']='1'
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def run(params):
    pprint(params)
    import algorithm.DRL4Route.train as DRL4Route_model
    DRL4Route_model.main(params)

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    from my_utils.utils import  dict_merge

    params = vars(get_params())
    params['pre_train'] = True #if False, please specify the path of the pretrained model
    params['model_path'] =  ''

    args_lst = []
    for model in ['DRL4Route_REINFORCE', 'DRL4Route_REINFORCE_GAE']:
        for hs in [64]:
            for rl_r in [0.3]:
                for trace_decay in [0.99]:
                    deeproute_params = {'model': model, 'hidden_size': hs, 'rl_ratio':rl_r, 'trace_decay': trace_decay}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)

    print(args_lst)
    for p in args_lst:
        run(p)
        print('finished!!!')








