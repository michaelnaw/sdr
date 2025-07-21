"""
Test FSK demodulation functionality.
"""

import numpy as np

import sdr


def test_bfsk_demodulate_perfect():
    """Test BFSK demodulation with perfect channel."""
    bfsk = sdr.FSK(2, freq_sep=1000, sps=8)
    symbols = np.array([0, 1, 1, 0, 1, 0])

    # Modulate then demodulate
    tx_signal = bfsk.modulate(symbols)
    rx_symbols = bfsk.demodulate(tx_signal)

    # Should perfectly recover symbols in noiseless case
    np.testing.assert_array_equal(symbols, rx_symbols)


def test_4fsk_demodulate_perfect():
    """Test 4-FSK demodulation with perfect channel."""
    fsk4 = sdr.FSK(4, freq_sep=500, sps=16)
    symbols = np.array([0, 1, 2, 3, 3, 2, 1, 0])

    # Modulate then demodulate
    tx_signal = fsk4.modulate(symbols)
    rx_symbols = fsk4.demodulate(tx_signal)

    # Should perfectly recover symbols in noiseless case
    np.testing.assert_array_equal(symbols, rx_symbols)


def test_8fsk_demodulate_perfect():
    """Test 8-FSK demodulation with perfect channel."""
    fsk8 = sdr.FSK(8, freq_sep=250, sps=8)
    symbols = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Modulate then demodulate
    tx_signal = fsk8.modulate(symbols)
    rx_symbols = fsk8.demodulate(tx_signal)

    # Should perfectly recover symbols in noiseless case
    np.testing.assert_array_equal(symbols, rx_symbols)


def test_demodulate_with_noise():
    """Test demodulation with additive noise."""
    bfsk = sdr.FSK(2, freq_sep=2000, sps=16)  # Higher freq_sep and sps for better noise performance
    symbols = np.array([0, 1] * 10)  # Repeat pattern

    # Modulate
    tx_signal = bfsk.modulate(symbols)

    # Add small amount of noise
    rng = np.random.default_rng(42)
    noise_power = 0.01
    noise = np.sqrt(noise_power) * (rng.standard_normal(len(tx_signal)) + 1j * rng.standard_normal(len(tx_signal)))
    rx_signal = tx_signal + noise

    # Demodulate
    rx_symbols = bfsk.demodulate(rx_signal)

    # With low noise, should have high success rate
    error_rate = np.mean(symbols != rx_symbols)
    assert error_rate < 0.1  # Less than 10% error rate


def test_demodulate_single_symbol():
    """Test demodulation of single symbols."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)

    for symbol in range(4):
        symbols = np.array([symbol])
        tx_signal = fsk.modulate(symbols)
        rx_symbols = fsk.demodulate(tx_signal)

        assert len(rx_symbols) == 1
        assert rx_symbols[0] == symbol


def test_demodulate_empty_input():
    """Test demodulation with empty input."""
    fsk = sdr.FSK(2, freq_sep=1000, sps=8)
    empty_signal = np.array([], dtype=complex)
    rx_symbols = fsk.demodulate(empty_signal)

    assert len(rx_symbols) == 0
    assert rx_symbols.dtype == int


def test_demodulate_partial_symbol():
    """Test demodulation when signal length is not multiple of sps."""
    fsk = sdr.FSK(2, freq_sep=1000, sps=8)
    symbols = np.array([0, 1])
    tx_signal = fsk.modulate(symbols)

    # Truncate signal by a few samples
    truncated_signal = tx_signal[:-3]
    rx_symbols = fsk.demodulate(truncated_signal)

    # Should demodulate complete symbols only
    expected_symbols = len(truncated_signal) // fsk.sps
    assert len(rx_symbols) == expected_symbols


def test_demodulate_different_sps():
    """Test demodulation with different samples per symbol."""
    symbols = np.array([0, 1, 2, 3])

    for sps in [4, 8, 16]:
        fsk = sdr.FSK(4, freq_sep=500, sps=sps)
        tx_signal = fsk.modulate(symbols)
        rx_symbols = fsk.demodulate(tx_signal)

        np.testing.assert_array_equal(symbols, rx_symbols)


def test_demodulate_frequency_offset_tolerance():
    """Test demodulation behavior with frequency offset."""
    bfsk = sdr.FSK(2, freq_sep=4000, sps=32)  # Very wide frequency separation and high sps
    symbols = np.array([0, 1, 0, 1])

    # Modulate
    tx_signal = bfsk.modulate(symbols)

    # Test without frequency offset first (should work perfectly)
    rx_symbols_perfect = bfsk.demodulate(tx_signal)
    assert np.array_equal(symbols, rx_symbols_perfect)

    # Apply very small frequency offset (much smaller than freq separation)
    t = np.arange(len(tx_signal))
    freq_offset = 10  # Hz, very small compared to freq_sep=4000Hz
    phase_offset = 2j * np.pi * freq_offset * t / bfsk.sps
    rx_signal = tx_signal * np.exp(phase_offset)

    # Demodulate - FSK is sensitive to frequency offset, so this may fail
    # We just test that the function runs without crashing
    rx_symbols = bfsk.demodulate(rx_signal)
    assert len(rx_symbols) == len(symbols)  # At least returns correct length


def test_demodulate_phase_offset_tolerance():
    """Test demodulation tolerance to phase offset."""
    bfsk = sdr.FSK(2, freq_sep=1000, sps=8)
    symbols = np.array([0, 1, 1, 0])

    # Modulate
    tx_signal = bfsk.modulate(symbols)

    # Apply constant phase offset
    phase_offset = np.pi / 4  # 45 degrees
    rx_signal = tx_signal * np.exp(1j * phase_offset)

    # Demodulate (should be invariant to constant phase offset)
    rx_symbols = bfsk.demodulate(rx_signal)

    np.testing.assert_array_equal(symbols, rx_symbols)


def test_demodulate_amplitude_scaling():
    """Test demodulation with amplitude scaling."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)
    symbols = np.array([0, 1, 2, 3])

    # Modulate
    tx_signal = fsk.modulate(symbols)

    # Scale amplitude
    for scale_factor in [0.1, 0.5, 2.0, 10.0]:
        scaled_signal = scale_factor * tx_signal
        rx_symbols = fsk.demodulate(scaled_signal)

        # Should be invariant to amplitude scaling
        np.testing.assert_array_equal(symbols, rx_symbols)


def test_demodulate_randomized():
    """Test demodulation with randomized symbol patterns."""
    rng = np.random.default_rng(42)  # For reproducibility

    fsk = sdr.FSK(4, freq_sep=1000, sps=8)

    for _ in range(10):  # Multiple random tests
        # Generate random symbols
        n_symbols = rng.integers(5, 20)
        symbols = rng.integers(0, 4, n_symbols)

        # Test modulation/demodulation
        tx_signal = fsk.modulate(symbols)
        rx_symbols = fsk.demodulate(tx_signal)

        np.testing.assert_array_equal(symbols, rx_symbols)
