from pathlib import Path
from omegaconf import DictConfig

import hydra
import torch
import torch.nn as nn

import torchaudio.transforms as TT
from utils.file import load_config
from robpitch.pitch.base.modules.pitch_predictor import ConvPitchPredictor


class RobPitch(nn.Module):
    """Noise and reverberation robust pitch predictor"""

    def __init__(
        self,
        predictor: nn.Module = ConvPitchPredictor(),
        **kwargs,
    ):
        super().__init__()
        self.predictor = predictor

    @classmethod
    def load_from_checkpoint(
        cls, config_path: Path, ckpt_path: Path, device: torch.device, **kwargs
    ):
        """
        Load pre-trained model

        Args:
            config_path (Path): path to the model model configuration.
            ckpt_path (Path): path of model checkpoint.
            device (torch.device): The device to load the model onto.

        Returns:
            model (nn.Module): The loaded model instance.
        """
        cfg = load_config(config_path)
        if "config" in cfg.keys():
            cfg = cfg["config"]
        cls.device = device
        cls.init_mel_transformer(cls, cfg, device)
        predictor = hydra.utils.instantiate(cfg["model"]["pitch_model"]["predictor"])
        model = cls(predictor=predictor)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict["pitch_model"], strict=False
        )
        for key in missing_keys:
            print("missing tensor {}".format(key))
        for key in unexpected_keys:
            print("unexpected tensor {}".format(key))
        model.eval()
        model.to(device)
        return model

    def init_mel_transformer(self, config, device):
        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

    def forward(self, x: torch.Tensor):
        """
        Pitch predictor forward

        Args:
            x (torch.Tensor): # [B, T]
        """
        mel = self.mel_transformer(x)

        pitch_logits, _, latent = self.predictor(mel)

        return {
            "pitch": torch.argmax(pitch_logits, dim=1),
            "latent": latent,
        }


# test
if __name__ == "__main__":
    from utils.audio import load_audio

    robpitch = RobPitch()
    device = torch.device("cuda:6")
    model = robpitch.load_from_checkpoint(
        config_path="config.yaml",
        ckpt_path="model.bin",
        device=device,
    )
    # Load and process the audio
    wav = load_audio(
        "/nfslocal/data1/xinsheng/data/vocoder/wavs/32k/P99_moyin_speaker_record_moyin_龚洪海_wav_split_1/龚洪海_降噪后_100_0.wav",
        sampling_rate=16000,
        volume_normalize=True,
    )
    wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

    outputs = model(wav)

    print(outputs)
