import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_activation_steganography_extract_ecc_decode(tmp_path):
    """
    End-to-end CLI test: extract thresholded bits and (optionally) ECC-decode them.

    This does not claim real neuron-level actuation; it validates the artifact schema
    and that Hamming(7,4) decoding is wired when the user opts in.
    """
    repo_root = Path(__file__).resolve().parents[1]
    out_prefix = tmp_path / "stegext_"

    # Payload 1011 encoded via Hamming(7,4) -> codeword: 0 1 1 0 0 1 1
    code_bits = [0, 1, 1, 0, 0, 1, 1]
    activations = np.array([[(-1.0 if b == 0 else 1.0) for b in code_bits]], dtype=np.float32)
    act_path = tmp_path / "activations.npy"
    np.save(act_path, activations)

    cmd = [
        sys.executable,
        "-m",
        "neurinspectre.cli.main",
        "activation-steganography",
        "extract",
        "--activations",
        str(act_path),
        "--target-neurons",
        "0,1,2,3,4,5,6",
        "--threshold",
        "0.0",
        "--ecc-decode",
        "--ecc-pad-bits",
        "0",
        "--out-prefix",
        str(out_prefix),
    ]
    run = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert run.returncode == 0, f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"

    out_path = Path(f"{out_prefix}steg_extract.json")
    assert out_path.exists()
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out.get("bits") == code_bits
    assert out.get("decoded_bits") == [1, 0, 1, 1]
    assert (out.get("ecc") or {}).get("scheme") == "hamming74"

