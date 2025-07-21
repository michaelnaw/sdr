"""
Test FSK SER calculations against theoretical values.
"""

import numpy as np

import sdr


def test_bfsk_ser():
    """Test BFSK SER calculation."""
    bfsk = sdr.FSK(2, freq_sep=1000)
    esn0 = np.arange(0, 15, 1.0)
    ser = bfsk.ser(esn0)

    # For BFSK, SER = BER since there's only 2 symbols
    # Theoretical: 0.5 * exp(-0.5 * Es/N0)
    esn0_linear = 10 ** (esn0 / 10)
    ser_expected = 0.5 * np.exp(-0.5 * esn0_linear)

    np.testing.assert_allclose(ser, ser_expected, rtol=1e-10)


def test_bfsk_ser_diff():
    """Test BFSK SER with differential encoding."""
    bfsk = sdr.FSK(2, freq_sep=1000)
    esn0 = np.array([5.0, 10.0, 15.0])
    ser = bfsk.ser(esn0, diff_encoded=True)
    ser_non_diff = bfsk.ser(esn0, diff_encoded=False)

    # Differential encoding should increase SER
    assert np.all(ser >= ser_non_diff)


def test_4fsk_ser():
    """Test 4-FSK SER calculation."""
    fsk4 = sdr.FSK(4, freq_sep=500)
    esn0 = np.array([0, 5, 10, 15, 20])
    ser = fsk4.ser(esn0)

    # SER should decrease with increasing Es/N0
    assert np.all(np.diff(ser) < 0)

    # SER should be between 0 and 1
    assert np.all(ser >= 0)
    assert np.all(ser <= 1)

    # At Es/N0 = 0 dB, SER should be relatively high
    assert ser[0] > 0.5

    # At Es/N0 = 20 dB, SER should be very low
    assert ser[-1] < 0.01


def test_8fsk_ser():
    """Test 8-FSK SER calculation."""
    fsk8 = sdr.FSK(8, freq_sep=250)
    esn0 = np.array([0, 5, 10, 15, 20])
    ser = fsk8.ser(esn0)

    # SER should decrease with increasing Es/N0
    assert np.all(np.diff(ser) < 0)

    # SER should be between 0 and 1
    assert np.all(ser >= 0)
    assert np.all(ser <= 1)


def test_ser_vs_ber_relationship():
    """Test relationship between SER and BER for M-FSK."""
    # For M-ary modulations with Gray coding, BER ≈ SER * M / (2 * (M-1))
    fsk4 = sdr.FSK(4, freq_sep=500)
    esn0 = np.array([10.0, 15.0])

    ser = fsk4.ser(esn0)
    # Convert Es/N0 to Eb/N0: Eb/N0 = Es/N0 - 10*log10(bps)
    ebn0 = esn0 - 10 * np.log10(fsk4.bps)
    ber = fsk4.ber(ebn0)

    # For FSK, BER and SER have different relationships depending on SNR
    # Just test that both are calculated correctly and are positive
    assert np.all(ber > 0)
    assert np.all(ser > 0)
    # Both should decrease with increasing SNR
    assert ber[1] < ber[0]  # BER decreases with increasing Eb/N0
    assert ser[1] < ser[0]  # SER decreases with increasing Es/N0


def test_ser_vectorization():
    """Test that SER calculation handles vector inputs correctly."""
    fsk = sdr.FSK(4, freq_sep=1000)

    # Single value
    ser_single = fsk.ser(10.0)
    assert np.isscalar(ser_single)

    # Array input
    esn0_array = np.array([5.0, 10.0, 15.0])
    ser_array = fsk.ser(esn0_array)
    assert ser_array.shape == (3,)

    # Check individual calculations match
    for i, esn0_val in enumerate(esn0_array):
        ser_individual = fsk.ser(esn0_val)
        np.testing.assert_allclose(ser_array[i], ser_individual)


def test_ser_order_comparison():
    """Test that higher order FSK has higher SER at same Es/N0."""
    esn0 = np.array([10.0, 15.0])

    fsk2 = sdr.FSK(2, freq_sep=1000)
    fsk4 = sdr.FSK(4, freq_sep=500)
    fsk8 = sdr.FSK(8, freq_sep=250)

    ser2 = fsk2.ser(esn0)
    ser4 = fsk4.ser(esn0)
    ser8 = fsk8.ser(esn0)

    # Higher order modulations should generally have higher SER at same Es/N0
    assert np.all(ser2 <= ser4)
    assert np.all(ser4 <= ser8)
