# RobPitch

## Overview

RobPitch is a pitch detection model trained to be robust against noise and reverberation environments. The model has been trained on 1600 hours of high-quality data, supplemented by an equivalent amount of simulated noisy and reverberant data, ensuring effective performance under challenging acoustic conditions.


## Usage Example

```
import librosa
import torch
import numpy as np

from robpitch import RobPitch
from utils.audio import load_audio

# Initialize the model
robpitch = RobPitch()
device = torch.device("cuda:6")

# Load model from checkpoint
model = robpitch.load_from_checkpoint(
    config_path="config.yaml",
    ckpt_path="model.bin",
    device=device
)

# Load and process the audio
wav = load_audio(
    "path/to/audio",
    sampling_rate=16000,
    volume_normalize=True
)
wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

# Get model outputs
outputs = model(wav)
pitch = outputs['pitch']
latent_feature = outputs['latent']

```

For more detailed usage examples, refer to the exp/demo.ipynb notebook.

## Todo List

- [ ] Upload the pre-trained model for easy access.

- [ ] Enable pip install.