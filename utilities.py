import os
import numpy as np
import torch
from PIL import Image
from time import localtime, strftime

gLoggerFile = None

def setupLogger(fpath):
    fileMode = 'w'
    input = None
    while input is None:
        print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
        # input = io.read()
        input = 'o'
        if input == 'o':
            fileMode = 'w'
        elif input == 'a':
            fileMode = 'a'
        elif input == 'q':
            os.exit()
        else:
            break
    global gLoggerFile
    gLoggerFile = open(fpath, fileMode)


def shutdownLogger():
    if gLoggerFile is not None:
        gLoggerFile.close()


def logging(message, mute=False):
    timeStamp = strftime("%Y-%m-%d %H:%M:%S", localtime())
    msgFormatted = '[{}]  {}'.format(timeStamp, message)
    if not mute:
        print(msgFormatted)
    if gLoggerFile is not None:
        gLoggerFile.write(msgFormatted + '\n')
        gLoggerFile.flush()


def modelSize(model):
    params = model.parameters()

    count = 0
    countForEach = []
    for i, a_param in enumerate(params):
        nParam = a_param.numel()
        count = count + nParam
        countForEach.append(nParam)
    return count, torch.LongTensor(countForEach)


def diagnoseGradients(params):
    """
    [[ Diagnose gradients by checking the value range and the ratio of the norms
    ARGS:
      - `params`     : first arg returned by net:parameters()
      - `gradParams` : second arg returned by net:parameters()
    ]]
    """
    pass
    # for param in params:
    #     print(type(param.data), param.size())


def checkpoint(model, epoch=None):
    package = {
        'epoch': epoch if epoch else 'N/A',
        'state_dict': model.state_dict(),
    }
    return package


# TODO: fix me
def modelState(model):
    """
    [[ Get model state, including model parameters (weights and biases) and
         running mean/var in batch normalization layers
    ARGS:
      - `model` : network model
    RETURN:
      - `state` : table, model states
    ]]
    """
    parameters = model.parameters()
    bnVars = []
    bnLayers = model.findModules('nn.BatchNormalization')
    for i in xrange(len(bnLayers)):
        bnVars[2 * i] = bnLayers[i].running_mean
        bnVars[2 * i + 1] = bnLayers[i].running_var

    bnLayers = model.findModules('nn.SpatialBatchNormalization')
    for i in xrange(len(bnLayers)):
        bnVars[2 * i] = bnLayers[i].running_mean
        bnVars[2 * i + 1] = bnLayers[i].running_var

    state = {'parameters' : parameters, 'bnVars' : bnVars}
    return state


def loadModelState(model, stateToLoad):
    state = modelState(model)
    assert len(state.parameters) == len(stateToLoad.parameters)
    assert len(state.bnVars) == len(stateToLoad.bnVars)
    for i in xrange(len(state.parameters)):
        state.parameters[i] = stateToLoad.parameters[i]
    for i in xrange(len(state.bnVars)):
        state.bnVars[i] = stateToLoad.bnVars[i]


def loadAndResizeImage(imagePath):
    img = Image.open(imagePath)
    img = img.convert('YCbCr')
    img = np.asarray(img)
    img = img[:, :, 0]
    img = Image.fromarray(img)
    img = img.resize((100, 32))
    img = np.asarray(img, dtype=float)
    return img