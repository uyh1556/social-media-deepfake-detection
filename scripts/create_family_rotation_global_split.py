#!/usr/bin/env python3
"""Build the FF-only global split for the cyclic family-rotation study."""

from __future__ import annotations

import create_family_coverage_global_split as base


base.PROTOCOL_NAME = "family_rotation_global_source_aware_v1"
base.METHODS = {
    "simswap": ("SimSwap", "FS", "rotation_candidate", "pair"),
    "blendface": ("BlendFace", "FS", "rotation_candidate", "pair"),
    "inswap": ("InSwapper", "FS", "rotation_candidate", "pair"),
    "facedancer": ("FaceDancer", "FS", "rotation_candidate", "pair"),
    "fsgan": ("FSGAN", "FS", "rotation_candidate", "pair"),
    "e4s": ("e4s", "FS", "rotation_candidate", "pair"),
    "wav2lip": (
        "Wav2Lip", "FR", "rotation_candidate", "appearance_external_driver"
    ),
    "fomm": ("FOMM", "FR", "rotation_candidate", "driver_appearance"),
    "sadtalker": (
        "SadTalker", "FR", "rotation_candidate", "appearance_external_driver"
    ),
    "hyperreenact": (
        "HyperReenact", "FR", "rotation_candidate", "driver_appearance"
    ),
    "tpsm": ("TPSM", "FR", "rotation_candidate", "driver_appearance"),
    "pirender": ("PIRender", "FR", "rotation_candidate", "driver_appearance"),
    "stylegan3": ("StyleGAN3", "EFS", "rotation_candidate", "latent_seed"),
    "dit": ("DiT", "EFS", "rotation_candidate", "latent_seed"),
    "styleganxl": ("StyleGAN-XL", "EFS", "rotation_candidate", "latent_seed"),
    "pixart": ("PixArt-alpha", "EFS", "rotation_candidate", "pixart_sample"),
    "vqgan": ("VQGAN", "EFS", "rotation_candidate", "latent_seed"),
    "sd21": ("SD2.1", "EFS", "rotation_candidate", "pixart_sample"),
}
base.PLANNED_MISSING_METHODS = ()


if __name__ == "__main__":
    base.main()
