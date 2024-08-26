# Copyright (c) 2024 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description:
    This script contains a collection of functions designed to handle various
    audio processing.
"""

from pathlib import Path
import random
import numpy as np
import soxr
import soundfile


def audio_volume_normalize(audio, coeff=0.2):
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
):
    r"""Load audio file with target sampling rate and lsength

    Args:
        adfile (Path): path to audio file.
        sampling_rate (int, optional): target sampling rate. Defaults to None.
        length (int, optional): target audio length. Defaults to None.
        volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
        segment_duration (int): random select a segment with duration of {segment_duration}s.
                                Defualt to None which means the whole audio will be used.

    Returns:
        audio (np.ndarray): audio
    """

    audio, sr = soundfile.read(adfile)

    # print('audio', adfile, audio)
    audio = np.array(audio.squeeze())

    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate

    if segment_duration is not None:
        seg_length = int(sr * segment_duration)
        audio = random_select_audio_segment(audio, seg_length)

    # Audio volume normalize
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    # check the audio length
    if length is not None:
        assert abs(audio.shape[0] - length) < 1000
        if audio.shape[0] > length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, int(length - audio.shape[0])))
    return audio


def random_select_audio_segment(audio: np.ndarray, length: int):
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]
