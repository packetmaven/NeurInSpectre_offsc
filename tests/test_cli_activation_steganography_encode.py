import json
import subprocess
import sys
from pathlib import Path


def test_activation_steganography_encode_uses_ecc(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_prefix = tmp_path / "stegenc_"

    cmd = [
        sys.executable,
        "-m",
        "neurinspectre.cli.main",
        "activation-steganography",
        "encode",
        "--model",
        "dummy",
        "--tokenizer",
        "dummy",
        "--prompt",
        "Hello",
        "--payload-bits",
        "1,0,1",
        "--target-neurons",
        "0,1,2",
        "--out-prefix",
        str(out_prefix),
    ]
    run = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert run.returncode == 0, f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"

    meta_path = Path(f"{out_prefix}steg_metadata.json")
    prompt_path = Path(f"{out_prefix}encoded_prompt.txt")
    assert meta_path.exists()
    assert prompt_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("method") == "ecc"

    encoded = prompt_path.read_text(encoding="utf-8", errors="ignore")
    assert "STEG_ECC" in encoded

