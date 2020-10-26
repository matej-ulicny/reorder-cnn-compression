import argparse
import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import numba as nb
from scipy.fftpack import dct

@nb.jit(nopython=True, parallel=True)
def dist_calc(max_elem, X, query, distances):
    for i in nb.prange(query.shape[0]):
        #distances[i] = 1 - (np.dot(X[:, max_elem], X[:, query[i]]) / (np.linalg.norm(X[:, max_elem]) * np.linalg.norm(X[:, query[i]]))) # cosine distance
        distances[i] = np.sum(np.square(X[:, max_elem] - X[:, query[i]])) # euclidean distance
    return distances

@nb.jit(nopython=True, parallel=True)
def mag_calc(X, magnitudes):
    for i in nb.prange(X.shape[1]):
        magnitudes[i] = np.sum(np.square(X[:, i]))
    return magnitudes

def argsort(X):
    timer = time.time()
    ind = np.zeros(X.shape[1], dtype=np.int)
    query = np.array(list(range(0, X.shape[1])))	    
    magnitudes = np.zeros(X.shape[1], dtype=np.float32)
    magnitudes = mag_calc(X, magnitudes)
    ind[0] = np.argmax(magnitudes)
    query = np.delete(query, ind[0])
    for i in range(1, X.shape[1]):
        distances = np.zeros(query.shape, dtype=np.float32)
        distances = dist_calc(ind[i-1], X, query, distances)
        min_ind = np.argmin(distances)
        ind[i] = query[min_ind]
        query = np.delete(query, min_ind)
    print('time: ', time.time() - timer)
    return ind


def in_name(lst, name):
    for item in lst:
        if item in name:
            return True
	return False

model_names = ['resnet50', 'mobilenet_v2']
modes = ['uniform', 'progressive']
	
parser = argparse.ArgumentParser(description='Compression script')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: '+' | '.join(model_names)+' (default: resnet50)')
parser.add_argument('--mode', default='uniform', choices=modes, help='compression mode: '+' | '.join(modes)+' (default: uniform)')
parser.add_argument('-g', '--groups', default=4, type=int)
parser.add_argument('-r', '--compression-rate', default=2.0, type=float)
parser.add_argument('--out', default='', type=str, metavar='PATH', help='name of the output file (default: compressed-arch-mode-g-r.pth)')

args = parser.parse_args()

if args.out = '':
    args.out = 'compressed-'+args.arch+'-'+args.mode+'-'+str(args.g)+'-'+str(args.r)+'.pth'
						 
model = models.__dict__[args.arch](pretrained=True)

dict = {}

for name, param in model.state_dict().items():
    # filtering out tensors that are not going to be compressed
    if 'resnet' in args.arch and 'fc.weight' not in name and (len(param.data.size()) != 4 or param.data.size(2) > 3 or param.data.size(3) != 1 or
	        'mobilenet_v2' in args.arch and 'classifier.1.weight' not in name and (len(param.data.size()) != 4 or param.data.size(2)>1 or 
	        in_name('features.7','features.6','features.5','features.4','features.3','features.2','features.1','features.0'], name)):
        dict.update({name: p})
        continue
    print(name)
	
    W = param.data.detach().numpy()
    W.resize((args.g, np.prod(W.shape) // args.g)) # reshape
    ind = argsort(W) 
    W = W[:,ind] # reorder
    ind = np.array([np.where(ind==i)[0][0] for i in range(len(ind))]) # reverse index
    dict.update({name.replace('weight', 'index'): nn.Parameter(torch.IntTensor(ind), requires_grad=False)}) # save index
 
	factor = math.sqrt(np.prod(X.shape)) / math.sqrt(64*64 if 'resnet' in args.arch else 384*64) # proportion of the tensor size with the reference
 
    r = args.r if args.uniform else 1 + factor * args.r # layer's compression rate
    t = int(W.shape[1] / r) # number of frequencies used
    Y = np.zeros((W.shape[0], t), dtype=np.float32)
    for i in range(W.shape[0]):
        Y[i] = dct(W[i], norm='ortho')[:t] # DCT of each row
    
    param.data = torch.Tensor(Y)
    dict.update({name: param})

torch.save(dict, args.out)