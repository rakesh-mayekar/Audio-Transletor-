"""Registry of installable Whisper models (CTranslate2 conversions of OpenAI Whisper).

Revisions are PINNED commit hashes, and model.bin carries a pinned SHA-256 so an install can be
verified against a known value rather than whatever a server returns. Licence: MIT (OpenAI
Whisper weights; Systran conversions). All are MULTILINGUAL checkpoints (no ".en" variants).
Whisper "turbo" is deliberately absent: it was not trained for the translate task.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelFileSpec:
    name: str
    size: int
    sha256: str | None = None  # None = recorded at install (trust-on-first-use) + size-checked


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    files: tuple[ModelFileSpec, ...]
    license: str = "MIT"
    parameters_m: int = 0
    note: str = ""

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def _spec(
    name: str,
    rev: str,
    params: int,
    cfg_size: int,
    bin_size: int,
    bin_sha: str,
    extra_sha: dict[str, str] | None = None,
) -> ModelSpec:
    x = extra_sha or {}
    return ModelSpec(
        name=name,
        repo_id=f"Systran/faster-whisper-{name}",
        revision=rev,
        parameters_m=params,
        files=(
            ModelFileSpec("config.json", cfg_size, x.get("config.json")),
            ModelFileSpec("model.bin", bin_size, bin_sha),
            ModelFileSpec("tokenizer.json", 2203239, x.get("tokenizer.json")),
            ModelFileSpec("vocabulary.txt", 459861, x.get("vocabulary.txt")),
        ),
    )


REGISTRY: dict[str, ModelSpec] = {
    "base": _spec(
        "base",
        "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        74,
        2309,
        145217532,
        "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
    ),
    "small": _spec(
        "small",
        "536b0662742c02347bc0e980a01041f333bce120",
        244,
        2370,
        483546902,
        "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
        {
            "config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
            "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
            "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
        },
    ),
    "medium": _spec(
        "medium",
        "08e178d48790749d25932bbc082711ddcfdfbc4f",
        769,
        2257,
        1527906378,
        "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae",
    ),
}
# "large-v3" is accepted by config but NOT installable until its revision/hash are pinned+tested.
