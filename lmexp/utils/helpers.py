import torch as t
import matplotlib.pyplot as plt

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

def get_a_b_probs(logits, a_token_id, b_token_id):
    if not isinstance(logits, t.Tensor):
        raise TypeError(f"Expected logits to be a torch.Tensor, but got {type(logits)}")
    
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to be a 3D tensor, but got shape {logits.shape}")
    
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob

def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'