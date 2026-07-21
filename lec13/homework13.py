import numpy as np
import librosa


def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frame
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''

    frames = np.array([
        speech[m:m + frame_length]
        for m in range(
            0,
            len(speech) - frame_length,
            frame_skip
        )
    ])

    A = librosa.lpc(
        frames,
        order=order
    )

    nframes, frame_length = frames.shape

    excitation = np.zeros(
        (nframes, frame_length)
    )

    for frame in range(nframes):

        for samp in range(order, frame_length):

            excitation[frame, samp] = frames[frame, samp]

            for k in range(1, order + 1):

                excitation[frame, samp] += (
                    A[frame, k] *
                    frames[frame, samp - k]
                )

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frame
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''

    synthesis = np.zeros(
        len(e)
    )

    nframes, orderp1 = A.shape

    order = orderp1 - 1

    for n in range(len(synthesis)):

        synthesis[n] = e[n]

        frame = min(
            n // frame_skip,
            nframes - 1
        )

        for k in range(1, order + 1):

            if n - k >= 0:

                synthesis[n] -= (
                    A[frame, k] *
                    synthesis[n - k]
                )

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''

    gain = np.sqrt(
        np.sum(
            excitation ** 2,
            axis=1
        )
    )

    nframes = len(gain)

    e_robot = np.zeros(
        nframes * frame_skip
    )

    e_robot[::T0] = 1

    for n in range(nframes):

        e_robot[
            n * frame_skip:
            (n + 1) * frame_skip
        ] *= gain[n]

    return gain, e_robot