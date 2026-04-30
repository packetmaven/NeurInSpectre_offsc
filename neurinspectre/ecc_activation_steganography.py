"""
ECC-backed activation steganography (artifact-grade implementation).

This module exists to replace the marker-only fallback in
`neurinspectre.activation_steganography`.

Design notes:
- The repo's "activation steganography" CLI currently *encodes* a payload into a
  prompt string. It does not (yet) guarantee neuron-level actuation in a given
  LLM; the "activation" part is an experimental interface.
- The ECC layer is still useful: it makes the payload robust to noisy extraction
  (e.g., threshold flips when reading activations).

We implement a simple Hamming(7,4) code (single-bit error correction) with a
human-readable marker format appended to the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


def _as_bit_list(bits: Sequence[int] | Iterable[int]) -> List[int]:
    out: List[int] = []
    for b in list(bits):
        try:
            v = int(b)
        except Exception:
            v = 0
        out.append(1 if v != 0 else 0)
    return out


def _bits_to_str(bits: Sequence[int]) -> str:
    return "".join("1" if int(b) else "0" for b in bits)


def _hamming74_encode_nibble(d: Sequence[int]) -> List[int]:
    """
    Encode 4 data bits to a 7-bit Hamming(7,4) codeword.

    Bit order (classic):
      positions: 1 2 3 4 5 6 7
                 p1 p2 d1 p3 d2 d3 d4
    """
    if len(d) != 4:
        raise ValueError("Hamming(7,4) expects 4 data bits")
    d1, d2, d3, d4 = (1 if int(x) else 0 for x in d)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def hamming74_encode(bits: Sequence[int]) -> tuple[List[int], int]:
    """
    Encode arbitrary-length payload bits using Hamming(7,4).

    Returns:
        (encoded_bits, pad_bits)
    """
    data = _as_bit_list(bits)
    pad = (-len(data)) % 4
    if pad:
        data = list(data) + [0] * pad
    out: List[int] = []
    for i in range(0, len(data), 4):
        out.extend(_hamming74_encode_nibble(data[i : i + 4]))
    return out, int(pad)


def _hamming74_syndrome(cw: Sequence[int]) -> int:
    """
    Return the 1-indexed error position indicated by the Hamming(7,4) syndrome.

    0 means "no error detected".
    """
    if len(cw) != 7:
        raise ValueError("Hamming(7,4) codeword must have length 7")
    b = [1 if int(x) else 0 for x in cw]
    # Bit order matches `_hamming74_encode_nibble`:
    #   positions 1..7: p1 p2 d1 p3 d2 d3 d4
    p1, p2, d1, p3, d2, d3, d4 = b
    s1 = p1 ^ d1 ^ d2 ^ d4  # parity over {1,3,5,7}
    s2 = p2 ^ d1 ^ d3 ^ d4  # parity over {2,3,6,7}
    s3 = p3 ^ d2 ^ d3 ^ d4  # parity over {4,5,6,7}
    # Syndrome bits map to the parity-bit positions (1,2,4).
    return int(s1 + 2 * s2 + 4 * s3)


def _hamming74_decode_codeword(cw: Sequence[int]) -> tuple[List[int], bool, int]:
    """
    Decode a single 7-bit codeword into 4 data bits.

    Returns:
        (data_bits, corrected, error_pos)
    """
    bits = [1 if int(x) else 0 for x in cw]
    err = _hamming74_syndrome(bits)
    corrected = False
    if err != 0:
        # Correct the indicated bit (1-indexed position).
        idx = int(err) - 1
        if 0 <= idx < 7:
            bits[idx] ^= 1
            corrected = True
    # Extract data bits: positions 3,5,6,7 -> indices 2,4,5,6
    data = [bits[2], bits[4], bits[5], bits[6]]
    return data, corrected, int(err)


def hamming74_decode(encoded_bits: Sequence[int], *, pad_bits: int = 0) -> Dict[str, Any]:
    """
    Decode a Hamming(7,4)-encoded bitstream.

    This is primarily used by the `activation-steganography extract` CLI when the
    caller opts into ECC decoding.

    Args:
        encoded_bits: The extracted *code bits* (length should be a multiple of 7).
        pad_bits: Number of zero pad bits that were appended to the data bits
            during encode to reach a multiple of 4.

    Returns:
        Dict containing decoded bits and decode diagnostics.
    """
    code = _as_bit_list(encoded_bits or [])
    pad = int(pad_bits or 0)
    if pad < 0 or pad > 3:
        raise ValueError("pad_bits must be in [0, 3]")

    n_codewords = int(len(code) // 7)
    leftover = int(len(code) - 7 * n_codewords)
    decoded: List[int] = []
    corrected_codewords = 0
    error_positions: List[int] = []

    for i in range(n_codewords):
        cw = code[i * 7 : (i + 1) * 7]
        data, corrected, err = _hamming74_decode_codeword(cw)
        decoded.extend(data)
        if corrected:
            corrected_codewords += 1
        error_positions.append(int(err))

    if pad:
        decoded = decoded[:-pad] if pad <= len(decoded) else []

    return {
        "scheme": "hamming74",
        "pad_bits": int(pad),
        "decoded_bits": [int(b) for b in decoded],
        "code_bits": [int(b) for b in code],
        "codewords": int(n_codewords),
        "leftover_code_bits": int(leftover),
        "corrected_codewords": int(corrected_codewords),
        # For auditability: syndrome-reported positions per codeword (0 means "no error").
        "error_positions": [int(x) for x in error_positions],
    }


def _marker(
    *,
    scheme: str,
    payload_bits: Sequence[int],
    encoded_bits: Sequence[int],
    pad_bits: int,
    target_neurons: Sequence[int],
) -> str:
    neurons = ",".join(str(int(n)) for n in list(target_neurons))
    return (
        "[STEG_ECC:v1"
        f"|scheme={scheme}"
        f"|data={_bits_to_str(payload_bits)}"
        f"|code={_bits_to_str(encoded_bits)}"
        f"|pad={int(pad_bits)}"
        f"|neurons={neurons}"
        "]"
    )


@dataclass(frozen=True)
class steganographyresult:
    prompt: str
    encoded_prompt: str
    payload_bits: List[int]
    encoded_bits: List[int]
    target_neurons: List[int]
    scheme: str
    pad_bits: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": str(self.scheme),
            "pad_bits": int(self.pad_bits),
            "payload_bits": [int(b) for b in self.payload_bits],
            "encoded_bits": [int(b) for b in self.encoded_bits],
            "target_neurons": [int(n) for n in self.target_neurons],
            "prompt_len": int(len(self.prompt)),
            "encoded_prompt_len": int(len(self.encoded_prompt)),
        }


class eccactivationsteganography:
    """
    Public entrypoint used by `neurinspectre.activation_steganography`.
    """

    def __init__(self, *, scheme: str = "hamming74"):
        self.scheme = str(scheme).strip().lower()
        if self.scheme not in {"hamming74"}:
            raise ValueError(f"Unsupported ECC scheme: {scheme}")

    def encode_payload(self, prompt: str, payload_bits: List[int], target_neurons: List[int]) -> str:
        """
        Return a prompt with an appended ECC marker.

        The marker is explicit by design so reviewers can verify what was encoded.
        """
        base = "" if prompt is None else str(prompt)
        bits = _as_bit_list(payload_bits or [])
        neurons = [int(n) for n in list(target_neurons or [])]
        if not bits or not neurons:
            return base

        encoded, pad = hamming74_encode(bits)
        mark = _marker(
            scheme=self.scheme,
            payload_bits=bits,
            encoded_bits=encoded,
            pad_bits=pad,
            target_neurons=neurons,
        )
        # Keep the prompt readable; separate marker with a space.
        return f"{base} {mark}"

    def encode_payload_result(self, prompt: str, payload_bits: List[int], target_neurons: List[int]) -> steganographyresult:
        bits = _as_bit_list(payload_bits or [])
        neurons = [int(n) for n in list(target_neurons or [])]
        encoded, pad = hamming74_encode(bits) if bits else ([], 0)
        encoded_prompt = self.encode_payload(prompt, payload_bits, target_neurons)
        return steganographyresult(
            prompt=str(prompt or ""),
            encoded_prompt=str(encoded_prompt),
            payload_bits=list(bits),
            encoded_bits=list(encoded),
            target_neurons=list(neurons),
            scheme=str(self.scheme),
            pad_bits=int(pad),
        )

