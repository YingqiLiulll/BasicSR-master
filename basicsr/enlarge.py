import argparse
import torch
#import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from models import FSRCNN

parser = argparse.ArgumentParser()
parser.add_argument('--weights-file', type=str, required=True)
parser.add_argument('--scale', type=int, default=3)
args = parser.parse_args()
#cudnn.benchmark = True
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FSRCNN(scale_factor=args.scale)#.to(device)
state_dict = model.state_dict()
for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

#model.eval()

plt.figure(figsize=(8, 8))
conv1 = dict(model.first_part.named_children())['0']
localw1 = conv1.weight.cpu().clone()
#print("total of number of filter : ", len(localw1))
b1=torch.max(localw1)
c1=torch.min(localw1)
for i in range(0, len(localw1)):
    localw0 = (localw1[i]-c1)/abs(b1-c1)
    # print(localw0.shape)
    # mean of 3 channel.
    # localw0 = torch.mean(localw0,dim=0)
    # there should be 3(3 channels) 11 * 11 filter.
    plt.subplot(7, 8, i + 1)
    plt.axis('off')
    plt.imshow(localw0[0, :, :].detach(), cmap='gray',vmin=0, vmax=1)
#plt.savefig('.FIR.png')
plt.show()

plt.figure(figsize=(10, 10))
conv2 = model.last_part
localw2 = conv2.weight.cpu().clone()
#print("total of number of filter : ", len(localw2))
#a=torch.zeros([1,9,9])
#for i in range(0, len(localw2)):
#    a+=localw2[i]
b=torch.max(localw2)
c=torch.min(localw2)

print(b,c)
for i in range(0, len(localw2)):
    #localwl = localw2[i]
    localwl = (localw2[i]-c)/abs(b-c)
    # print(localw0.shape)
    # mean of 3 channel.
    # localw0 = torch.mean(localw0,dim=0)
    # there should be 3(3 channels) 11 * 11 filter.
    plt.subplot(7,8, i + 1)
    plt.axis('off')

    plt.imshow(localwl[0, :, :].detach(), cmap='gray',vmin=0, vmax=1)
plt.savefig('.OUT1.png')
plt.show()
