import os
import torch
from config import get_config
from utilities import loadModelState, loadAndResizeImage, logging, modelSize
from torch.utils.serialization import load_lua
from torch import nn
from models.siamese import Siamese

load_from_torch7 = False

print('Loading model...')
model_dir = 'models/snapshot/'
model_load_path = os.path.join(model_dir, 'snapshot_epoch_1.pt')
gConfig = get_config()
gConfig.model_dir = model_dir

criterion = nn.HingeEmbeddingLoss()
model = Siamese()

package = torch.load(model_load_path)

model.load_state_dict(package['state_dict'])
model.eval()
print('Model loaded from {}'.format(model_load_path))

logging('Model configuration:\n{}'.format(model))

modelSize, nParamsEachLayer = modelSize(model)
logging('Model size: {}\n{}'.format(modelSize, nParamsEachLayer))

params = model.parameters()

for i, a_param in enumerate(params):
    print a_param

exit(0)

imagePath = '../data/demo.png'
img = loadAndResizeImage(imagePath)
text, raw = recognizeImageLexiconFree(model, img)
print('Recognized text: {} (raw: {})'.format(text, raw))
