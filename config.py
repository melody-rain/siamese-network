from torch import optim


def get_config():
    class Config():
        displayInterval = 100
        testInterval = 1000
        nTestDisplay = 15
        trainBatchSize = 64
        valBatchSize = 256
        snapshotInterval = 10000
        maxIterations = 2000000
        optimizer = optim.SGD
        optim_config = {}
        train_set_path = 'data'
        val_set_path = 'data'

    return Config()