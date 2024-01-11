import os as os_ori
os_ori.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy
from einops import rearrange

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
import warnings
warnings.filterwarnings("ignore")

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS
    from utils.freeze_model import freeze_model

    distributed = False

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f'Number of params before freezed: {n_parameters}')
    if cfg.get('freeze_dict', False):
        print(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
        freeze_model(my_model, cfg.freeze_dict)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f'Number of params after freezed: {n_parameters}')

    my_model = my_model.cuda()
    raw_model = my_model

    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)
        
    # resume and load
    print('work dir: ' + args.work_dir)
    ckpt = torch.load(cfg.load_from, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    raw_model.load_state_dict(state_dict, strict=False)
    print(f'loading pretrained model from: {cfg.load_from} complete!')
    
    epoch = 0
    my_model.eval()
    os.environ['eval'] = 'true'
    
    with torch.no_grad():
        for i_iter_val, (input_occs, _, metas) in enumerate(val_dataset_loader):
            with open(args.work_dir + f'visual_input_{i_iter_val}.npy', 'wb') as f:
                np.save(f, input_occs.numpy())

            input_occs = input_occs.cuda()            
            result_dict = my_model(input_occs)
            output = result_dict['sem_pred'].cpu().numpy()

            with open(args.work_dir + f'visual_output_{i_iter_val}.npy', 'wb') as f:
                np.save(f, output)
            break

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
        