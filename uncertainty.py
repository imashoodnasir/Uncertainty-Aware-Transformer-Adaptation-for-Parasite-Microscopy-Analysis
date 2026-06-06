import torch
import torch.nn.functional as F

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

@torch.no_grad()
def mc_classification_predict(model, x, samples=10):
    model.eval()
    enable_dropout(model)
    probs = []
    for _ in range(samples):
        logits = model(x)
        probs.append(F.softmax(logits, dim=1))
    stack = torch.stack(probs, dim=0)
    mean = stack.mean(dim=0)
    var = stack.var(dim=0)
    return mean, var

@torch.no_grad()
def mc_segmentation_predict(model, x, samples=10):
    model.eval()
    enable_dropout(model)
    probs, vars_ = [], []
    for _ in range(samples):
        logits, log_var = model(x)
        probs.append(torch.sigmoid(logits))
        vars_.append(torch.exp(log_var))
    probs = torch.stack(probs, dim=0)
    vars_ = torch.stack(vars_, dim=0)
    mean = probs.mean(dim=0)
    epistemic = probs.var(dim=0)
    aleatoric = vars_.mean(dim=0)
    total = epistemic + aleatoric
    return mean, epistemic, aleatoric, total
