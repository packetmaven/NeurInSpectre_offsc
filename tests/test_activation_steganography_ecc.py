from neurinspectre.activation_steganography import ActivationSteganography, ECC_AVAILABLE


def test_activation_steganography_ecc_available_and_encodes():
    # The audit previously documented a marker-only fallback. This test enforces that
    # the ECC implementation exists and the import path is wired correctly.
    assert ECC_AVAILABLE is True

    steg = ActivationSteganography()
    out = steg.encode_payload("Hello", [1, 0, 1, 1], [0, 1, 2, 3])
    assert isinstance(out, str)
    assert "STEG_ECC" in out
    assert "scheme=hamming74" in out
    assert "data=1011" in out

