import json
import logging
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import pygmo as pg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AIW_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        
        upp, low = preds[:, [0]], preds[:, [1]]
        target = target[:, 0]
        width = (upp - low).abs()
        width_mean = width.mean()
        width_sc = width_mean/(target.max() - target.min())
        
        loss = width_sc
        
        return loss


class AIW_loss_NOST(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        
        upp, low = preds[:, [0]], preds[:, [1]]
        width = (upp - low).abs()
        width_mean = width.mean()
        
        loss = width_mean
        
        return loss


class PICP_alpha_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        upp, low = preds[:, 0], preds[:, 1]
        target = target[:, 0]
        
        soften_ = 160
        K_SU = torch.sigmoid(soften_ * (upp - target))
        K_SL = torch.sigmoid(soften_ * (target - low))
        K_S = torch.mul(K_SU, K_SL)
        
        PICP_S = torch.mean(K_S)
        
        loss = 1-PICP_S
        
        return loss


class PICP_alpha_loss_hard(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target, device):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        upp, low = preds[:, 0], preds[:, 1]
        target = target[:, 0]
        
        K_HU = torch.max(torch.tensor(0).to(device),torch.sign(upp - target))
        K_HL = torch.max(torch.tensor(0).to(device),torch.sign(target - low))
        K_H = torch.mul(K_HU, K_HL)
        
        PICP_H = torch.mean(K_H)
        
        loss = 1-PICP_H
        
        return loss

def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")


def save_args(folder, args, name='config.json', check_exists=False):
    set_logger()
    path = Path(folder)
    if check_exists:
        if path.exists():
            logging.warning(f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(
        vars(args),
        open(path / name, "w")
    )


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles) 
    y = np.sin(angles)
    return np.c_[x, y]



### HV approximation 
def pseu_hv(front_alphas, front):
    suma = sum([front.loc[np.argmin(abs(front['task2_loss'] - alpha))]['task1_loss'] for alpha in front_alphas])
    hv = front_alphas.max() - front_alphas.min() - suma * (front_alphas.max() - front_alphas.min())/len(front_alphas)
    return hv




@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = AIW_loss()
    loss2 = PICP_alpha_loss()
    results = defaultdict(list)
    
    for ray in rays:
        total = 0.
        l1, l2 = 0., 0.
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()
        
        for x_val, y_val in loader:
            hypernet.zero_grad()
            x_val = x_val.to(device)
            y_val  = y_val.to(device)
            bs = len(y_val)
            weights = hypernet(ray)
            low, up = targetnet(x_val, weights)
            p_int = torch.cat((low, up), axis=1)
            curr_l1 = loss1(p_int, y_val[:, [0]])
            curr_l2 = loss2(p_int, y_val[:, [0]])
            l1 += curr_l1 * bs
            l2 += curr_l2 * bs
            total += bs
            
        results['ray'].append(ray.squeeze(0).cpu().numpy().tolist())
        results['task1_loss'].append(l1.cpu().item() / total)
        results['task2_loss'].append(l2.cpu().item() / total)
        
    d = {key: results[key] for key in results.keys() & {'task1_loss', 'task2_loss'}}
    df_loss = pd.DataFrame.from_dict(d)
    
    ref = [1.00, 1.00]
    hyp = pg.hypervolume(df_loss.values)
    hyp_val = hyp.compute(ref)
    
    #front_alphas = np.linspace(start=0, stop=1, num=500)
    #front_alphas = pd.Series(front_alphas)
    #hyp_val = pseu_hv(front_alphas, df_loss)
    
    results['hypervolume'].append(hyp_val)
    
    return results, hyp_val



@torch.no_grad()
def test(hypernet, targetnet, val_dict, PINP, loader, rays, device):
    hypernet.eval()
    alpha = 1-PINP
    loss1 = AIW_loss_NOST()
    loss2 = PICP_alpha_loss_hard()
    l1, l2 = 0., 0.
    total = 0.
    results = defaultdict(list)
    
    losses = np.array(val_dict['task2_loss']) 
    cond = losses <= alpha
    if(np.any(cond)):
        losses = losses[cond]
        rays = np.array(val_dict['ray'])[cond]
    pos = np.argmin(np.abs((np.array(losses) - alpha)))
    pref = rays[pos]
    ray = torch.from_numpy(pref.astype(np.float32)).to(device)
    
    for x_test, y_test in loader:
        hypernet.zero_grad()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        bs = len(y_test)
        weights = hypernet(ray)
        low, up = targetnet(x_test, weights)
        p_int = torch.cat((low, up), axis=1)
        curr_l1 = loss1(p_int, y_test[:, [0]])
        curr_l2 = loss2(p_int, y_test[:, [0]], device)
        l1 += curr_l1 * bs
        l2 += curr_l2 * bs
        total += bs
    results['PICP'].append(1 - (l2.cpu().item() / total))
    results['AIW'].append(l1.cpu().item() / total)
    
    return results
