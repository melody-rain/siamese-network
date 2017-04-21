import torch
from torchvision import transforms
import argparse
from training import train_model
from config import get_config
from models.siamese import Siamese
from torch import nn
from dataset import SiameseDataset
from utilities import setupLogger, logging
import os

parser = argparse.ArgumentParser(description='CRNN')
parser.add_argument('--model-dir', type=str, dest='model_dir',
                    help='path to where models are saved', default='models/snapshot/')
parser.add_argument('--load-path', dest='load_path', type=str,
                    help='path to pretrained model', default='')
parser.add_argument('--cuda', dest='cuda', action="store_true",
                    help='Use cuda to train model', default=False)
parser.add_argument('--gpu-id', type=int, dest='gpu_id',
                    help='Set GPU id', default=0)
parser.add_argument('--batch-size', type=int, dest='batch_size',
                    help='batch size', default=32)
args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    SiameseDataset('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    SiameseDataset('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True, **kwargs)

g_config = get_config()

model_dir = args.model_dir
setupLogger(os.path.join(model_dir, 'log.txt'))
g_config.model_dir = model_dir

criterion = nn.HingeEmbeddingLoss()
model = Siamese()

# load model snapshot
load_path = args.load_path
if load_path is not '':
    snapshot = torch.load(load_path)
    # loadModelState(model, snapshot)
    model.load_state_dict(snapshot['state_dict'])
    logging('Model loaded from {}'.format(load_path))

train_model(model, criterion, train_loader, test_loader, g_config, use_cuda=False)