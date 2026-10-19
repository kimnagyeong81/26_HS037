# -*- coding: utf-8 -*-

"""
cnn_ensemble.py

Merry AI - VGG19 CNN Ensemble Inference

구조
    Audio
      ↓
    Mono
      ↓
    16 kHz Resample
      ↓
    10 sec Crop / Zero Padding
      ↓
    Mel Spectrogram
      ↓
    dB 변환
      ↓
    Z-score Normalization
      ↓
    224 x 224 Resize
      ↓
    3 Channel
      ↓
    VGG19 x 3
      ↓
    Soft Voting
      ↓
    Normal / MCI

이 파일은 모델을 최초 1회만 로드하고
predict(audio_path)를 반복 호출하는 형태로 사용한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.models as models
import soundfile as sf

# =============================================================================
# 1. 기본 설정
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

CLASS_NAMES = {
    0: "Normal",
    1: "MCI",
}

TARGET_SAMPLE_RATE = 16000
DURATION_SEC = 10

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

IMAGE_SIZE = 224


# =============================================================================
# 2. 모델 설정
# =============================================================================
#
# 실제 서버의 모델 저장 위치에 맞게 수정하면 됨.
#
# dropout은 학습 당시 설정을 맞춰 놓지만,
# eval() 상태에서는 Dropout이 비활성화되므로
# 실제 추론 결과에는 영향을 주지 않는다.
# =============================================================================

MODEL_DIR = Path(
    os.getenv(
        "CNN_MODEL_DIR",
        "/Server/models",
    )
)


DEFAULT_MODEL_CONFIGS = [
    {
        "name": "search_030",
        "path": str(
            MODEL_DIR
            / (
                "search_030_lr3e-05_bs16_dur10_mel128_"
                "wd0.002_freezelast_block_drop0.5_cw1.pth"
            )
        ),
        "dropout": 0.5,
        "weight": 1.0,
    },
    {
        "name": "search_044",
        "path": str(
            MODEL_DIR
            / (
                "search_044_lr1e-05_bs8_dur10_mel128_"
                "wd0.001_freezelast_block_drop0.6_cw1.pth"
            )
        ),
        "dropout": 0.6,
        "weight": 1.0,
    },
    {
        "name": "search_048",
        "path": str(
            MODEL_DIR
            / (
                "search_048_lr1e-05_bs8_dur10_mel128_"
                "wd0.002_freezelast_block_drop0.6_cw1.pth"
            )
        ),
        "dropout": 0.6,
        "weight": 1.0,
    },
]

# =============================================================================
# 3. VGG19 모델 생성
# =============================================================================

def build_vgg19_model(
    num_classes: int = NUM_CLASSES,
    dropout_p: float = 0.5,
) -> nn.Module:
    """
    학습 당시 사용한 VGG19 classifier 구조.

    기본 torchvision VGG19 classifier 대신:

        Linear(25088 -> 4096)
        ReLU
        Dropout
        Linear(4096 -> 1024)
        ReLU
        Dropout
        Linear(1024 -> 2)

    구조를 사용한다.
    """

    model = models.vgg19(
        weights=None
    )

    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(
            in_features,
            4096,
        ),
        nn.ReLU(
            inplace=True
        ),
        nn.Dropout(
            p=dropout_p
        ),

        nn.Linear(
            4096,
            1024,
        ),
        nn.ReLU(
            inplace=True
        ),
        nn.Dropout(
            p=dropout_p
        ),

        nn.Linear(
            1024,
            num_classes,
        ),
    )

    return model


# =============================================================================
# 4. Checkpoint 유틸
# =============================================================================

def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    다양한 checkpoint 저장 형식을 지원한다.

    지원 형태
        torch.save(model.state_dict())

        torch.save({
            "state_dict": model.state_dict()
        })

        torch.save({
            "model_state_dict": model.state_dict()
        })

        torch.save({
            "model": model.state_dict()
        })

        torch.save({
            "net": model.state_dict()
        })
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "checkpoint가 dict 형식이 아닙니다."
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):
        value = checkpoint.get(key)

        if isinstance(
            value,
            dict,
        ):
            return value

    # checkpoint 자체가 state_dict인 경우
    if checkpoint and all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):
        return checkpoint

    raise ValueError(
        "지원하지 않는 checkpoint 형식입니다."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    DataParallel 학습 시 붙는

        module.

    prefix 제거.
    """

    result = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        result[key] = value

    return result


def load_model(
    model_path: str,
    dropout_p: float,
    device: torch.device,
) -> nn.Module:
    """
    VGG19 생성 후 checkpoint 로드.
    """

    path = Path(
        model_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}"
        )

    print(
        f"[CNN] 모델 로딩 시작: {path.name}"
    )

    model = build_vgg19_model(
        num_classes=NUM_CLASSES,
        dropout_p=dropout_p,
    )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing_keys:
        print(
            f"[CNN][WARN] {path.name} "
            f"missing_keys:"
        )

        for key in missing_keys:
            print(
                f"    {key}"
            )

    if unexpected_keys:
        print(
            f"[CNN][WARN] {path.name} "
            f"unexpected_keys:"
        )

        for key in unexpected_keys:
            print(
                f"    {key}"
            )

    model.to(
        device
    )

    model.eval()

    print(
        f"[CNN] 모델 로딩 완료: {path.name}"
    )

    return model


# =============================================================================
# 5. Audio 전처리
# =============================================================================

def load_audio(
    audio_path: str,
) -> torch.Tensor:
    """
    1. 음성 로드
    2. Mono 변환
    3. 16 kHz resampling
    4. 10초 crop 또는 zero-padding

    반환:
        [1, 160000]
    """

    path = Path(
        audio_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"음성 파일이 없습니다: {path}"
        )

    audio_data, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if audio_data.size == 0:
        raise ValueError(
            f"비어 있는 음성 파일입니다: {path}"
        )

    # soundfile:
    #   [samples, channels]
    #
    # PyTorch / 기존 CNN 코드:
    #   [channels, samples]
    #
    # 형태로 변환한다.
    waveform = torch.from_numpy(
        audio_data.T
    ).contiguous()

    sample_rate = int(sample_rate)

    # -------------------------------------------------------------------------
    # Stereo -> Mono
    # -------------------------------------------------------------------------

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # -------------------------------------------------------------------------
    # Resample -> 16 kHz
    # -------------------------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
        )

        waveform = resampler(
            waveform
        )

    # -------------------------------------------------------------------------
    # 10초 고정
    # -------------------------------------------------------------------------

    target_length = (
        TARGET_SAMPLE_RATE
        * DURATION_SEC
    )

    current_length = (
        waveform.shape[1]
    )

    if current_length > target_length:

        waveform = waveform[
            :,
            :target_length,
        ]

    elif current_length < target_length:

        padding = (
            target_length
            - current_length
        )

        waveform = F.pad(
            waveform,
            (
                0,
                padding,
            ),
        )

    return waveform


def waveform_to_mel_image(
    waveform: torch.Tensor,
) -> torch.Tensor:
    """
    waveform을 학습 당시와 동일한
    VGG19 입력 tensor로 변환한다.

    반환:
        [1, 3, 224, 224]
    """

    # -------------------------------------------------------------------------
    # Mel Spectrogram
    # -------------------------------------------------------------------------

    mel_transform = (
        torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
    )

    mel = mel_transform(
        waveform
    )

    # -------------------------------------------------------------------------
    # Amplitude -> dB
    # -------------------------------------------------------------------------

    amplitude_to_db = (
        torchaudio.transforms.AmplitudeToDB()
    )

    mel_db = amplitude_to_db(
        mel
    )

    # -------------------------------------------------------------------------
    # Z-score normalization
    # -------------------------------------------------------------------------

    mean = mel_db.mean()
    std = mel_db.std()

    mel_db = (
        mel_db - mean
    ) / (
        std + 1e-8
    )

    # -------------------------------------------------------------------------
    # [1, mel, time]
    #
    # ↓
    #
    # [1, 1, mel, time]
    # -------------------------------------------------------------------------

    mel_db = mel_db.unsqueeze(
        0
    )

    # -------------------------------------------------------------------------
    # Resize -> 224 x 224
    # -------------------------------------------------------------------------

    mel_db = F.interpolate(
        mel_db,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # -------------------------------------------------------------------------
    # 1 channel -> 3 channel
    #
    # VGG19 input:
    # [B, 3, 224, 224]
    # -------------------------------------------------------------------------

    mel_db = mel_db.repeat(
        1,
        3,
        1,
        1,
    )

    return mel_db


def preprocess_audio(
    audio_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    최종 CNN input tensor 생성.
    """

    waveform = load_audio(
        audio_path
    )

    input_tensor = (
        waveform_to_mel_image(
            waveform
        )
    )

    return input_tensor.to(
        device
    )


# =============================================================================
# 6. 개별 모델 추론
# =============================================================================

@torch.inference_mode()
def predict_one_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """
    단일 VGG19 모델 추론.
    """

    logits = model(
        input_tensor
    )

    probs = torch.softmax(
        logits,
        dim=1,
    )

    pred_idx = int(
        torch.argmax(
            probs,
            dim=1,
        ).item()
    )

    return {
        "pred_idx": pred_idx,
        "pred_label": (
            CLASS_NAMES[
                pred_idx
            ]
        ),
        "normal_prob": float(
            probs[
                0,
                0,
            ].item()
        ),
        "mci_prob": float(
            probs[
                0,
                1,
            ].item()
        ),
        "probs": (
            probs
            .detach()
            .cpu()
        ),
    }


# =============================================================================
# 7. Ensemble 계산
# =============================================================================

def soft_voting(
    results: list[dict],
) -> dict:
    """
    각 모델 probability의 단순 평균.

    최종 Merry AI 기본 판단 방식.
    """

    if not results:
        raise ValueError(
            "앙상블 결과가 없습니다."
        )

    probabilities = torch.cat(
        [
            result["probs"]
            for result in results
        ],
        dim=0,
    )

    mean_probs = (
        probabilities.mean(
            dim=0
        )
    )

    pred_idx = int(
        torch.argmax(
            mean_probs
        ).item()
    )

    return {
        "method": "soft_voting",
        "pred_idx": pred_idx,
        "pred_label": CLASS_NAMES[
            pred_idx
        ],
        "normal_prob": float(
            mean_probs[0].item()
        ),
        "mci_prob": float(
            mean_probs[1].item()
        ),
    }


def weighted_soft_voting(
    results: list[dict],
    weights: list[float],
) -> dict:
    """
    모델별 가중치를 적용한 soft voting.
    """

    if not results:
        raise ValueError(
            "앙상블 결과가 없습니다."
        )

    if len(results) != len(weights):
        raise ValueError(
            "모델 결과 개수와 weight 개수가 다릅니다."
        )

    probabilities = torch.cat(
        [
            result["probs"]
            for result in results
        ],
        dim=0,
    )

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    ).reshape(
        -1,
        1,
    )

    weighted_probs = (
        probabilities
        * weight_tensor
    )

    mean_probs = (
        weighted_probs.sum(
            dim=0
        )
        / weight_tensor.sum()
    )

    pred_idx = int(
        torch.argmax(
            mean_probs
        ).item()
    )

    return {
        "method": "weighted_soft_voting",
        "pred_idx": pred_idx,
        "pred_label": CLASS_NAMES[
            pred_idx
        ],
        "normal_prob": float(
            mean_probs[0].item()
        ),
        "mci_prob": float(
            mean_probs[1].item()
        ),
        "weights": weights,
    }


def hard_voting(
    results: list[dict],
) -> dict:
    """
    각 모델의 최종 class를 이용한 다수결.
    """

    if not results:
        raise ValueError(
            "앙상블 결과가 없습니다."
        )

    votes = [
        result["pred_idx"]
        for result in results
    ]

    normal_votes = (
        votes.count(0)
    )

    mci_votes = (
        votes.count(1)
    )

    if mci_votes > normal_votes:
        pred_idx = 1
    else:
        pred_idx = 0

    return {
        "method": "hard_voting",
        "pred_idx": pred_idx,
        "pred_label": CLASS_NAMES[
            pred_idx
        ],
        "normal_votes": normal_votes,
        "mci_votes": mci_votes,
        "votes": votes,
    }


# =============================================================================
# 8. CNN Ensemble 클래스
# =============================================================================

class CNNEnsemble:
    """
    Merry AI CNN Ensemble.

    서버 시작 시:

        cnn = CNNEnsemble()

    발화 분석 시:

        result = cnn.predict("/data/audio/test.wav")

    형태로 사용한다.

    모델은 __init__ 시 한 번만 GPU에 로드된다.
    """

    def __init__(
        self,
        model_configs: list[dict] | None = None,
        device: torch.device | None = None,
    ):
        self.device = (
            device
            if device is not None
            else DEVICE
        )

        self.model_configs = (
            model_configs
            if model_configs is not None
            else DEFAULT_MODEL_CONFIGS
        )

        self.models: list[nn.Module] = []
        self.valid_configs: list[dict] = []

        print(
            "=" * 60
        )

        print(
            "[CNN] Merry AI Ensemble 초기화"
        )

        print(
            f"[CNN] Device: {self.device}"
        )

        print(
            "=" * 60
        )

        self._load_models()

    # =========================================================================

    def _load_models(
        self,
    ) -> None:

        for config in self.model_configs:

            path = config[
                "path"
            ]

            dropout = float(
                config.get(
                    "dropout",
                    0.5,
                )
            )

            try:
                model = load_model(
                    model_path=path,
                    dropout_p=dropout,
                    device=self.device,
                )

                self.models.append(
                    model
                )

                self.valid_configs.append(
                    config
                )

            except Exception as exc:
                print(
                    "[CNN][ERROR] "
                    f"모델 로딩 실패: {path}"
                )

                print(
                    f"[CNN][ERROR] {exc}"
                )

        if not self.models:
            raise RuntimeError(
                "로드 가능한 CNN 모델이 없습니다."
            )

        print(
            f"[CNN] 총 {len(self.models)}개 모델 로딩 완료"
        )

    # =========================================================================

    @torch.inference_mode()
    def predict(
        self,
        audio_path: str,
    ) -> dict:
        """
        하나의 음성 파일에 대해

        1. 한 번만 전처리
        2. 3개의 VGG19 추론
        3. Soft voting
        4. Weighted soft voting
        5. Hard voting

        을 수행한다.
        """

        input_tensor = (
            preprocess_audio(
                audio_path,
                self.device,
            )
        )

        individual_results = []

        # ---------------------------------------------------------------------
        # 각 모델 추론
        # ---------------------------------------------------------------------

        for index, model in enumerate(
            self.models
        ):
            result = predict_one_model(
                model,
                input_tensor,
            )

            config = (
                self.valid_configs[
                    index
                ]
            )

            individual_results.append(
                result
            )

            print(
                "[CNN] "
                f"{config['name']} -> "
                f"{result['pred_label']} "
                f"(Normal={result['normal_prob']:.4f}, "
                f"MCI={result['mci_prob']:.4f})"
            )

        # ---------------------------------------------------------------------
        # Soft Voting
        # ---------------------------------------------------------------------

        soft_result = soft_voting(
            individual_results
        )

        # ---------------------------------------------------------------------
        # Weighted Soft Voting
        # ---------------------------------------------------------------------

        weights = [
            float(
                config.get(
                    "weight",
                    1.0,
                )
            )
            for config in self.valid_configs
        ]

        weighted_result = (
            weighted_soft_voting(
                individual_results,
                weights,
            )
        )

        # ---------------------------------------------------------------------
        # Hard Voting
        # ---------------------------------------------------------------------

        hard_result = hard_voting(
            individual_results
        )

        # ---------------------------------------------------------------------
        # 최종 결과
        #
        # 기존 코드와 동일하게
        # 기본 최종 판단은 soft voting 사용
        # ---------------------------------------------------------------------

        result = {
            "pred_idx": (
                soft_result[
                    "pred_idx"
                ]
            ),
            "pred_label": (
                soft_result[
                    "pred_label"
                ]
            ),
            "normal_prob": (
                soft_result[
                    "normal_prob"
                ]
            ),
            "mci_prob": (
                soft_result[
                    "mci_prob"
                ]
            ),
            "ensemble_method": (
                "soft_voting"
            ),

            "soft_voting": (
                soft_result
            ),

            "weighted_soft_voting": (
                weighted_result
            ),

            "hard_voting": (
                hard_result
            ),

            "models": [],
        }

        # torch.Tensor는 JSON/MongoDB에 바로 넣을 수 없으므로
        # 필요한 값만 일반 Python 타입으로 변환
        for index, model_result in enumerate(
            individual_results
        ):

            config = (
                self.valid_configs[
                    index
                ]
            )

            result["models"].append(
                {
                    "name": config[
                        "name"
                    ],
                    "pred_idx": (
                        model_result[
                            "pred_idx"
                        ]
                    ),
                    "pred_label": (
                        model_result[
                            "pred_label"
                        ]
                    ),
                    "normal_prob": (
                        model_result[
                            "normal_prob"
                        ]
                    ),
                    "mci_prob": (
                        model_result[
                            "mci_prob"
                        ]
                    ),
                }
            )

        print(
            "[CNN] Ensemble -> "
            f"{result['pred_label']} "
            f"(Normal={result['normal_prob']:.4f}, "
            f"MCI={result['mci_prob']:.4f})"
        )

        return result

    # =========================================================================

    def close(
        self,
    ) -> None:
        """
        모델 메모리 해제.
        """

        self.models.clear()
        self.valid_configs.clear()

        if (
            self.device.type
            == "cuda"
        ):
            torch.cuda.empty_cache()

        print(
            "[CNN] Ensemble 종료"
        )


# =============================================================================
# 9. 단독 테스트
# =============================================================================

def main() -> None:

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Merry AI VGG19 CNN Ensemble"
        )
    )

    parser.add_argument(
        "audio",
        help="분석할 WAV/FLAC 파일 경로",
    )

    args = parser.parse_args()

    ensemble = None

    try:
        ensemble = CNNEnsemble()

        result = ensemble.predict(
            args.audio
        )

        print()
        print(
            "=" * 60
        )
        print(
            "CNN Ensemble Final Result"
        )
        print(
            "=" * 60
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        if ensemble is not None:
            ensemble.close()


if __name__ == "__main__":
    main()