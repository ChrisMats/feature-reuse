import torch
import numpy as np

def center_gram_numpy(gram):
    # FROM: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()
        
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)

    return gram

def center_gram_torch(gram):
    # FROM: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    if not torch.allclose(gram, gram.t()):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone().detach().double()

    n = gram.shape[0]
    gram.fill_diagonal_(0)
    means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
    means -= torch.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    gram.fill_diagonal_(0)

    return gram

def cka_numpy(x, y):
    # FROM: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    gram_x, gram_y = x.dot(x.T), y.dot(y.T)
    gram_x = center_gram_numpy(gram_x)
    gram_y = center_gram_numpy(gram_y)
    
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def cka_torch(x, y):
    # FROM: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    gram_x, gram_y = torch.mm(x, x.t()), torch.mm(y, y.t())

    gram_x = center_gram_torch(gram_x)
    gram_y = center_gram_torch(gram_y)

    scaled_hsic = torch.mm(gram_x.reshape(-1).unsqueeze(0), gram_y.reshape(-1).unsqueeze(1))[0][0]

    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def CKA(x, y):
    if isinstance(x, torch.Tensor):
        cka = cka_torch(x, y).item()
    elif isinstance(x, np.ndarray):
        cka = cka_numpy(x, y)
    else:
        raise ValueError(f"Type (type(x)) is not supported for CKA")
    return cka
