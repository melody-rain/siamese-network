import torch
import os
from utilities import logging, diagnoseGradients, checkpoint
from torch.autograd import Variable


def train_model(model, criterion, train_loader, test_loader, g_config, use_cuda=True):
    optimizer = g_config.optimizer(model.parameters(), lr=0.01, momentum=0.9)

    def train_batch(input_batch, target):
        model.train()
        input1, input2 = input_batch
        sim = target
        output0, output1, output = model(input1, input2)
        loss = criterion(output, sim)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss = loss / input1.size()[0]

        loss_sum = loss.data.sum()

        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            logging("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        return loss_value

    def validation(input_batch, target):
        model.eval()
        input1, input2 = input_batch
        sim = target
        output0, output1, output = model(input1, input2)
        # print torch.cat((output, target), 1)
        loss = criterion(output, sim)
        loss = loss / input1.size()[0]
        loss_sum = loss.data.sum()

        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            logging("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        # accuracy = 1.0 * nCorrect / nFrame
        logging('Test loss = {}, accuracy = {}'.format(loss_value, 0))

    # train loop
    avg_loss = 0
    epoch = 0
    while True:
        # validation
        for data0, data1, target in test_loader:
            if use_cuda:
                data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()
            data0, data1, target = Variable(data0, volatile=True), Variable(data1, volatile=True), Variable(target)
            validation((data0, data1), target)

        # train batch
        for batch_idx, (data0, data1, target) in enumerate(train_loader):
            if use_cuda:
                data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()
            data0, data1, target = Variable(data1), Variable(data0), Variable(target)
            avg_loss += train_batch((data0, data1), target)

            # display
            if batch_idx % g_config.displayInterval == 0:
                avg_loss = avg_loss / g_config.displayInterval
                logging('Batch {} - train loss = {}'.format(batch_idx, avg_loss))
                diagnoseGradients(model.parameters())
                avg_loss = 0

        # save snapshot
        save_path = os.path.join(g_config.model_dir, 'snapshot_epoch_{}.pt'.format(epoch))
        torch.save(checkpoint(model, epoch), save_path)
        logging('Snapshot saved to {}'.format(save_path))

        # terminate
        if epoch > g_config.maxIterations:
            logging('Maximum epoch reached, terminating ...')
            break

        epoch += 1

