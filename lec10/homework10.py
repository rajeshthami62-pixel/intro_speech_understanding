import numpy as np
import torch, torch.nn


def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label. Repeat each label five times.
    '''

    VAD_windowlen = int(0.025 * Fs)
    VAD_windowskip = int(0.010 * Fs)

    VAD_frames = np.array([
        waveform[m:m + VAD_windowlen]
        for m in range(
            0,
            len(waveform) - VAD_windowlen,
            VAD_windowskip
        )
    ])

    x_frames = np.array([
        waveform[m:m + int(0.004 * Fs)]
        for m in range(
            0,
            len(waveform) - int(0.004 * Fs),
            int(0.002 * Fs)
        )
    ])

    energy = np.sum(VAD_frames ** 2, axis=1)

    VAD = np.array([
        1 if energy[m] > 0.01 * np.max(energy) else 0
        for m in range(len(energy))
    ])

    startframes = [
        m for m in range(1, len(VAD))
        if VAD[m] == 1 and VAD[m - 1] == 0
    ]

    endframes = [
        m for m in range(1, len(VAD))
        if VAD[m] == 0 and VAD[m - 1] == 1
    ]

    labels = np.zeros(5 * len(VAD))

    for k in range(len(startframes)):
        labels[
            5 * startframes[k] : 5 * endframes[k]
        ] = k + 1

    mstft = np.abs(
        np.fft.fft(x_frames, axis=1)
    )

    features = 20 * np.log10(
        mstft[:, :mstft.shape[1] // 2] + 1e-10
    )

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels
    iterations (scalar) - number of iterations of training

    @return:
    model - a trained PyTorch neural network
    lossvalues - loss value for each iteration
    '''

    NFRAMES, NFEATS = features.shape

    NCLASSES = int(max(labels)) + 1

    model = torch.nn.Sequential(
        torch.nn.LayerNorm(NFEATS),
        torch.nn.Linear(NFEATS, NCLASSES)
    )

    lossfunction = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    lossvalues = np.zeros(iterations)

    for t in range(iterations):

        z = model(
            torch.tensor(
                features,
                dtype=torch.float
            )
        )

        loss = lossfunction(
            z,
            torch.tensor(
                labels,
                dtype=torch.long
            )
        )

        lossvalues[t] = loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array

    @return:
    probabilities (NFRAMES, NLABELS) - model output,
    transformed by softmax, detach().numpy().
    '''

    testresults = model(
        torch.tensor(
            features,
            dtype=torch.float
        )
    )

    testresults = torch.softmax(
        testresults,
        dim=1
    )

    return testresults.detach().numpy()