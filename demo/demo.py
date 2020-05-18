import os
import time

import numpy as np
from scipy.io import wavfile

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises,
    Mp3Compression,
)


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s
    """

    def __init__(self, description="Execution time"):
        self.description = description
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        print("{}: {:.3f} s".format(self.description, self.execution_time))


SAMPLE_RATE = 16000
CHANNELS = 1


def load_wav_file(sound_file_path):
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(sample_rate, SAMPLE_RATE)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        # ends up roughly between -1 and 1
        sound_np = np.divide(sound_np, 32767, dtype=np.float32)

    return sound_np


DEMO_DIR = os.path.dirname(__file__)

if __name__ == "__main__":
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder.
    """
    output_dir = os.path.join(DEMO_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    samples = load_wav_file(os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"))

    assert samples.dtype == np.float32

    # AddImpulseResponse
    augmenter = Compose(
        [AddImpulseResponse(p=1.0, ir_path=os.path.join(DEMO_DIR, "ir"))]
    )
    output_file_path = os.path.join(
        output_dir, "AddImpulseResponse_{:03d}.wav".format(0)
    )
    with timer("Apply AddImpulseResponse"):
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # FrequencyMask
    augmenter = Compose([FrequencyMask(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "FrequencyMask_{:03d}.wav".format(i)
        )
        with timer("Apply FrequencyMask"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeMask
    augmenter = Compose([TimeMask(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "TimeMask_{:03d}.wav".format(i))
        with timer("Apply TimeMask"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianSNR
    augmenter = Compose([AddGaussianSNR(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "AddGaussianSNR_{:03d}.wav".format(i)
        )
        with timer("Apply AddGaussianSNR"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianNoise
    augmenter = Compose(
        [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "AddGaussianNoise_{:03d}.wav".format(i)
        )
        with timer("Apply AddGaussianSNR"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeStretch
    augmenter = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "TimeStretch_{:03d}.wav".format(i))
        with timer("Apply TimeStretch"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # PitchShift
    augmenter = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "PitchShift_{:03d}.wav".format(i))
        with timer("Apply PitchShift"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift
    augmenter = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "Shift_{:03d}.wav".format(i))
        with timer("Apply Shift"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift without rollover
    augmenter = Compose(
        [Shift(min_fraction=-0.5, max_fraction=0.5, rollover=False, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "ShiftWithoutRollover_{:03d}.wav".format(i)
        )
        with timer("Apply ShiftWithoutRollover"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Normalize
    augmenter = Compose([Normalize(p=1.0)])
    output_file_path = os.path.join(output_dir, "Normalize_{:03d}.wav".format(0))
    with timer("Apply Normalize"):
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Resample
    augmenter = Compose([Resample(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "Resample_{:03d}.wav".format(i))
        with timer("Apply Resample"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # ClippingDistortion
    augmenter = Compose([ClippingDistortion(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "ClippingDistortion_{:03d}.wav".format(i)
        )
        with timer("Apply ClippingDistortion"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddBackgroundNoise
    augmenter = Compose(
        [
            AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
            )
        ]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "AddBackgroundNoise_{:03d}.wav".format(i)
        )
        with timer("Apply AddBackgroundNoise"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddShortNoises
    augmenter = Compose(
        [
            AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            )
        ]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "AddShortNoises_{:03d}.wav".format(i)
        )
        with timer("Apply AddShortNoises"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Mp3Compression
    augmenter = Compose([Mp3Compression(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "Mp3Compression_{:03d}.wav".format(i)
        )
        with timer("Apply Mp3Compression"):
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        print(augmenter.transforms[0].parameters)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
