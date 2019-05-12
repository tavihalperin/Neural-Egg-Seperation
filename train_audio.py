import museval
import os
import numpy as np
import torch
import torch.optim as optim
from models import Generator257
import torchvision.utils as vutils
import time
import utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def get_data(dataset):
    consts = utils.Constants(dataset)
    train = np.load('data/%s/%s.npz' % (dataset, 'train'))['spec']
    test_spec = np.load('data/%s/%s.npz' % (dataset, 'test'))['spec']
    len_train = train.shape[0]
    train_Y = np.abs(train[:len_train//2, 0:1])
    train_B = np.abs(train[len_train//2:, 1:2])
    train_B = train_B[:len(train_Y)]
    test_Y = np.abs(test_spec[:,0:1])
    test_B = np.abs(test_spec[:,1:2])
    print ('dataset shape', train_Y.shape, train_B.shape, test_spec.shape)
    return train_Y, train_B, test_Y, test_B, test_spec, consts

dataset = 'speech'
train_Y, train_B, test_Y, test_B, test_complex_spectrogram, consts = get_data(dataset)
model = Generator257(1, 1, 64).to(device)
criterion = torch.nn.L1Loss().to(device)

if not os.path.exists('ims_%s' % dataset):
    os.mkdir('ims_%s' % dataset)

num_iterations = 10
num_epochs = 25 # Every iteration this number of epochs of training
batchsize = 64

num_test_batches = test_Y.shape[0] // batchsize
test_Y = test_Y[:num_test_batches * batchsize]
test_B = test_B[:num_test_batches * batchsize]
test_complex_spectrogram = test_complex_spectrogram[:num_test_batches * batchsize]
B_test_ground_truth_audio = utils.spec2aud(test_complex_spectrogram[:, 1].transpose((1, 0, 2)).reshape((257, -1)), consts)
X_test_ground_truth_audio = utils.spec2aud(test_complex_spectrogram[:, 2].transpose((1, 0, 2)).reshape((257, -1)), consts)

estimated_X = train_Y * consts.naive_separation_multiplier

for iter in range(num_iterations):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for epoch in range(num_epochs):
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
    test_mask = []
    for i in range(num_test_batches):
        idx = i * batchsize + np.arange(batchsize)
        test_batch_Y = test_Y[idx]
        test_batch_B = test_B[idx]
        test_batch_Y = torch.from_numpy(test_batch_Y).to(device).float()
        test_batch_B = torch.from_numpy(test_batch_B).to(device).float()
        batch_mask = model(test_batch_Y)
        test_mask.append(batch_mask.cpu().data.numpy())

        loss = criterion(batch_mask * test_batch_Y, test_batch_B)
        test_error += loss.data.item()
    test_error /= num_test_batches

    # Visualize last test batch
    vutils.save_image(torch.log(test_batch_Y.data.abs() + 1),
                      'ims_%s/testY.png' % dataset, normalize=True)
    vutils.save_image(torch.log((batch_mask * test_batch_Y).data.abs() + 1),
                      'ims_%s/prediction_iter_%d.png' % (dataset, iter), normalize=True)
    vutils.save_image(torch.log(test_batch_B.data.abs() + 1),
                      'ims_%s/ground_truth.png' % dataset, normalize=True)

    test_mask = np.concatenate(test_mask, axis=0).squeeze()
    B_test_prediction_spectrogram = test_complex_spectrogram[:, 0] * test_mask
    X_test_prediction_spectrogram = test_complex_spectrogram[:, 0] * (1 - test_mask)

    B_pred_audio = utils.spec2aud(B_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)
    X_pred_audio = utils.spec2aud(X_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)
    X_metrics = museval.metrics.bss_eval(X_test_ground_truth_audio[None, :, None], X_pred_audio[None, :, None], window=consts.SR, hop=consts.SR)
    B_metrics = museval.metrics.bss_eval(B_test_ground_truth_audio[None, :, None], B_pred_audio[None, :, None], window=consts.SR, hop=consts.SR)

    sdr_X = np.median(np.nan_to_num(X_metrics[0].squeeze()))
    sdr_B = np.median(np.nan_to_num(B_metrics[0].squeeze()))

    print('\n%s iter %d done, train error %.5f, test error %.5f, time: %d seconds\n' % (dataset, iter, error, test_error, int(time.time() - start)))
    print('test sdr B %.2f, X %.2f' % (sdr_X, sdr_B))

    out_model = 'models/%s/nes' % dataset
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
