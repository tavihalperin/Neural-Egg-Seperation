import numpy as np
import torch
import torch.optim as optim
from models import Generator64
import time
import torchvision.utils as vutils
import os

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

def preprocess(x):
    x = x.transpose((0, 3, 1, 2))
    x = x / 255.0
    return x

def get_shoes_bags_dataset():
    shoes = preprocess(np.load('data/shoes/train.npy'))
    bags = preprocess(np.load('data/bags/train.npy'))
    train_B = shoes[:5000]
    train_X = bags[:5000]
    train_Y = shoes[5000:10000] + bags[5000:10000]
    test_B = preprocess(np.load('data/shoes/test.npy'))
    test_X = preprocess(np.load('data/bags/test.npy'))
    test_Y = test_B + test_X
    return train_Y, train_B, train_X, test_Y, test_B, test_X

train_Y, train_B, train_X, test_Y, test_B, test_X = get_shoes_bags_dataset()
model = Generator64(3, 3, 64).to(device)
criterion = torch.nn.L1Loss().to(device)

num_iterations = 10
num_epochs = 25
batchsize = 32

num_test_batches = test_Y.shape[0] // batchsize
test_Y = test_Y[:num_test_batches * batchsize]
test_B = test_B[:num_test_batches * batchsize]
test_X = test_X[:num_test_batches * batchsize]

estimated_X = train_Y / 2

for iter in range(num_iterations):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for epoch in range(num_epochs):
        nb = train_Y.shape[0] // batchsize
        print('iteration: %d, epoch: %d' % (iter, epoch))
        num_batches = train_Y.shape[0] // batchsize
        permutation_B = np.random.permutation(train_Y.shape[0])
        permutation_X = np.random.permutation(train_Y.shape[0])
        error = 0
        for i in range(num_batches):
            idx = i * batchsize + np.arange(batchsize)
            batch_B = train_B[permutation_B[idx]]
            batch_estimated_X = estimated_X[permutation_X[idx]]
            batch_B = torch.from_numpy(batch_B).to(device).float()
            batch_estimated_X = torch.from_numpy(batch_estimated_X).to(device).float()
            model.zero_grad()
            batch_mask = model(batch_B + batch_estimated_X)
            loss = criterion(batch_mask * (batch_B + batch_estimated_X), batch_B)
            loss.backward()
            optimizer.step()
            error += loss.data.item()
        error /= num_batches

    # Infer on test set
    test_error = 0
    for i in range(num_test_batches):
        idx = i * batchsize + np.arange(batchsize)
        test_batch_Y = test_Y[idx]
        test_batch_B = test_B[idx]
        test_batch_Y = torch.from_numpy(test_batch_Y).to(device).float()
        test_batch_B = torch.from_numpy(test_batch_B).to(device).float()
        batch_mask = model(test_batch_Y)

        loss = criterion(batch_mask * test_batch_Y, test_batch_B)
        test_error += loss.data.item()
    test_error /= num_test_batches

    print('\niter %d done, train error %.5f, test error %.5f, time: %d seconds\n' % (
        iter, error, test_error, int(time.time() - start)))

    if not os.path.exists('ims'):
        os.makedirs('ims')
    vutils.save_image(test_batch_Y.data / 2, 'ims/testY.png', normalize=False)
    vutils.save_image((batch_mask * test_batch_Y).data, 'ims/prediction_iter_%d.png' % iter,
                      normalize=False)
    vutils.save_image(test_batch_B.data, 'ims/ground_truth.png', normalize=True)

    out_model = 'models/shoes_bags/nes'
    if not os.path.exists(out_model):
        os.makedirs(out_model)
    torch.save(model, out_model + "/model-%d" % iter)

    # Estimate X with the new model
    num_batches = train_Y.shape[0] // batchsize
    for i in range(num_batches):
        idx = i * batchsize + np.arange(batchsize)
        Y_batch = train_Y[idx]
        Y_batch = torch.from_numpy(Y_batch).to(device).float()
        X_estimated_batch = (1 - model(Y_batch)) * Y_batch
        estimated_X[idx] = X_estimated_batch.cpu().data.numpy()

    Y_batch = train_Y[-batchsize:]
    Y_batch = torch.from_numpy(Y_batch).to(device).float()
    X_estimated_batch = (1 - model(Y_batch)) * Y_batch
    estimated_X[-batchsize:] = X_estimated_batch.cpu().data.numpy()