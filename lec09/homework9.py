import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    framelength = int(0.025 * Fs)
    frameskip = int(0.010 * Fs)

    frames = np.array([waveform[n:n+framelength]
                       for n in range(0, len(waveform)-framelength+1, frameskip)])

    energies = np.sum(frames**2, axis=1)
    e = 0.1 * np.max(energies)

    VAD = [1 if energies[m] > e else 0 for m in range(len(energies))]

    framestarts = [m for m in range(1, len(energies))
                   if VAD[m] == 1 and VAD[m-1] == 0]

    frameends = [m for m in range(1, len(energies))
                 if VAD[m] == 0 and VAD[m-1] == 1]

    segments = [waveform[framestarts[k]*frameskip:frameends[k]*frameskip+framelength]
                for k in range(min(len(framestarts), len(frameends)))]

    return segments


def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []

    framelength = int(0.004 * Fs)
    frameskip = int(0.002 * Fs)

    for k in range(len(segments)):
        waveform = segments[k]

        waveform = np.append(waveform[0], waveform[1:] - 0.95 * waveform[:-1])

        frames = np.array([waveform[n:n+framelength]
                           for n in range(0, len(waveform)-framelength+1, frameskip)])

        mstft = np.abs(np.fft.fft(frames, axis=1))

        lsgram = 20 * np.log10(np.maximum(mstft, 0.001 * np.amax(mstft)))

        models.append(np.mean(lsgram[:, :int(framelength//2)], axis=0))

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    segments = VAD(testspeech, Fs)

    sims = np.zeros((len(models), len(segments)))
    test_outputs = []

    framelength = int(0.004 * Fs)
    frameskip = int(0.002 * Fs)

    for k in range(len(segments)):
        x = segments[k]

        frames = np.array([x[m:m+framelength]
                           for m in range(0, len(x)-framelength+1, frameskip)])

        mstft = np.abs(np.fft.fft(frames, axis=1))

        sgram = 20 * np.log10(np.maximum(mstft, 0.001 * np.amax(mstft)))

        spectrum = np.mean(sgram[:, :int(framelength//2)], axis=0)

        for y in range(len(models)):
            sims[y, k] = np.dot(spectrum, models[y]) / (
                np.linalg.norm(spectrum) * np.linalg.norm(models[y])
            )

        test_outputs.append(labels[np.argmax(sims[:, k])])

    return sims, test_outputs