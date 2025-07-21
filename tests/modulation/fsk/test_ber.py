"""
Test FSK BER calculations against theoretical values.
"""

import numpy as np

import sdr


def test_bfsk_ber():
    """Test BFSK BER calculation."""
    bfsk = sdr.FSK(2, freq_sep=1000)
    ebn0 = np.arange(0, 15, 1.0)
    ber = bfsk.ber(ebn0)

    # Theoretical BFSK BER: 0.5 * exp(-0.5 * Eb/N0)
    ebn0_linear = 10 ** (ebn0 / 10)
    ber_expected = 0.5 * np.exp(-0.5 * ebn0_linear)

    np.testing.assert_allclose(ber, ber_expected, rtol=1e-10)


def test_bfsk_ber_diff():
    """Test BFSK BER with differential encoding."""
    bfsk = sdr.FSK(2, freq_sep=1000)
    ebn0 = np.array([5.0, 10.0, 15.0])
    ber = bfsk.ber(ebn0, diff_encoded=True)
    ber_non_diff = bfsk.ber(ebn0, diff_encoded=False)

    # Differential encoding should increase BER
    assert np.all(ber >= ber_non_diff)

    # For small error rates, BER_diff ≈ 2 * BER_non_diff
    ber_approx = 2 * ber_non_diff * (1 - ber_non_diff)
    np.testing.assert_allclose(ber, ber_approx, rtol=1e-10)


def test_4fsk_ber():
    """Test 4-FSK BER calculation."""
    fsk4 = sdr.FSK(4, freq_sep=500)
    ebn0 = np.array([0, 5, 10, 15])
    ber = fsk4.ber(ebn0)

    # BER should decrease with increasing Eb/N0
    assert np.all(np.diff(ber) < 0)

    # BER should be between 0 and 0.5
    assert np.all(ber >= 0)
    assert np.all(ber <= 0.5)


def test_8fsk_ber():
    """Test 8-FSK BER calculation."""
    fsk8 = sdr.FSK(8, freq_sep=250)
    ebn0 = np.array([0, 5, 10, 15])
    ber = fsk8.ber(ebn0)

    # BER should decrease with increasing Eb/N0
    assert np.all(np.diff(ber) < 0)

    # BER should be between 0 and 0.5
    assert np.all(ber >= 0)
    assert np.all(ber <= 0.5)


def test_ber_vectorization():
    """Test that BER calculation handles vector inputs correctly."""
    fsk = sdr.FSK(4, freq_sep=1000)

    # Single value
    ber_single = fsk.ber(10.0)
    assert np.isscalar(ber_single)

    # Array input
    ebn0_array = np.array([5.0, 10.0, 15.0])
    ber_array = fsk.ber(ebn0_array)
    assert ber_array.shape == (3,)

    # Check individual calculations match
    for i, ebn0_val in enumerate(ebn0_array):
        ber_individual = fsk.ber(ebn0_val)
        np.testing.assert_allclose(ber_array[i], ber_individual)


def test_ber_edge_cases():
    """Test BER calculation edge cases."""
    fsk = sdr.FSK(2, freq_sep=1000)

    # Very low Eb/N0
    ber_low = fsk.ber(-10.0)
    assert ber_low < 0.5
    assert ber_low > 0.4  # Should be close to 0.5 for very low SNR

    # Very high Eb/N0
    ber_high = fsk.ber(20.0)
    assert ber_high < 1e-5  # Should be very small for high SNR
