import numpy as np

# MixMatch 数据增强方法
def mixmatch(x1, x2, y1, y2, pseudo_labels, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    y_mixed_pseudo = lam * y1 + (1 - lam) * pseudo_labels
    return x_mixed, y_mixed_pseudo
