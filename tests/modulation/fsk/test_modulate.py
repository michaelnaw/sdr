"""
Test FSK modulation functionality.
"""

import numpy as np

import sdr


def test_bfsk_modulate_basic():
    """Test basic BFSK modulation."""
    bfsk = sdr.FSK(2, freq_sep=1000, sps=8)
    symbols = np.array([0, 1, 0, 1])
    tx_signal = bfsk.modulate(symbols)

    # Check output dimensions
    expected_length = len(symbols) * bfsk.sps
    assert len(tx_signal) == expected_length

    # Check output is complex
    assert tx_signal.dtype == complex


def test_4fsk_modulate_basic():
    """Test basic 4-FSK modulation."""
    fsk4 = sdr.FSK(4, freq_sep=500, sps=4)
    symbols = np.array([0, 1, 2, 3, 0])
    tx_signal = fsk4.modulate(symbols)

    # Check output dimensions
    expected_length = len(symbols) * fsk4.sps
    assert len(tx_signal) == expected_length

    # Check output is complex
    assert tx_signal.dtype == complex


def test_modulate_frequency_content():
    """Test that modulated signal contains expected frequency content."""
    bfsk = sdr.FSK(2, freq_sep=1000, sps=16)

    # Test single symbol modulation
    symbols_0 = np.array([0])
    tx_0 = bfsk.modulate(symbols_0)

    symbols_1 = np.array([1])
    tx_1 = bfsk.modulate(symbols_1)

    # Signals should be different
    assert not np.allclose(tx_0, tx_1)

    # Check that the signals have consistent magnitude (determined by pulse shape)
    # For rectangular pulse with sps=16, magnitude should be 1/sqrt(sps) ≈ 0.25
    expected_mag = 1.0 / np.sqrt(16)
    assert np.allclose(np.abs(tx_0), expected_mag, rtol=0.1)
    assert np.allclose(np.abs(tx_1), expected_mag, rtol=0.1)


def test_modulate_symbol_validation():
    """Test symbol validation in modulation."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)

    # Valid symbols
    valid_symbols = np.array([0, 1, 2, 3])
    tx_signal = fsk.modulate(valid_symbols)
    assert len(tx_signal) == len(valid_symbols) * fsk.sps

    # Invalid symbols are handled by validation in modulate()
    # Note: Out-of-range symbols are expected to fail


def test_modulate_empty_input():
    """Test modulation with empty input."""
    fsk = sdr.FSK(2, freq_sep=1000, sps=8)
    symbols = np.array([], dtype=int)
    tx_signal = fsk.modulate(symbols)

    assert len(tx_signal) == 0
    assert tx_signal.dtype == complex


def test_modulate_single_symbol():
    """Test modulation with single symbol."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)

    for symbol in range(4):
        symbols = np.array([symbol])
        tx_signal = fsk.modulate(symbols)

        assert len(tx_signal) == fsk.sps
        assert tx_signal.dtype == complex
        # Signal should have magnitude determined by pulse shape
        expected_mag = 1.0 / np.sqrt(fsk.sps)
        assert np.allclose(np.abs(tx_signal), expected_mag, rtol=0.1)


def test_modulate_different_sps():
    """Test modulation with different samples per symbol."""
    symbols = np.array([0, 1, 2, 3])

    for sps in [1, 4, 8, 16]:
        fsk = sdr.FSK(4, freq_sep=500, sps=sps)
        tx_signal = fsk.modulate(symbols)

        expected_length = len(symbols) * sps
        assert len(tx_signal) == expected_length
        assert tx_signal.dtype == complex


def test_modulate_frequency_separation():
    """Test modulation with different frequency separations."""
    symbols = np.array([0, 1])
    sps = 8

    for freq_sep in [100, 500, 1000, 2000]:
        bfsk = sdr.FSK(2, freq_sep=freq_sep, sps=sps)
        tx_signal = bfsk.modulate(symbols)

        assert len(tx_signal) == len(symbols) * sps
        assert tx_signal.dtype == complex
        # Signal should have magnitude determined by pulse shape
        expected_mag = 1.0 / np.sqrt(bfsk.sps)
        assert np.allclose(np.abs(tx_signal), expected_mag, rtol=0.1)


def test_map_symbols():
    """Test symbol mapping functionality."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)
    symbols = np.array([0, 1, 2, 3])

    # Test map_symbols method
    complex_symbols = fsk.map_symbols(symbols)

    # Should return complex array with proper length
    expected_length = len(symbols) * fsk.sps
    assert len(complex_symbols) == expected_length
    assert complex_symbols.dtype == complex

    # Should have unit magnitude
    assert np.allclose(np.abs(complex_symbols), 1.0, rtol=0.1)


def test_freq_map_property():
    """Test frequency mapping property."""
    # Test BFSK
    bfsk = sdr.FSK(2, freq_sep=1000)
    expected_freqs = np.array([-500.0, 500.0])  # Centered around 0
    np.testing.assert_allclose(bfsk.freq_map, expected_freqs)

    # Test 4-FSK
    fsk4 = sdr.FSK(4, freq_sep=500)
    expected_freqs = np.array([-750.0, -250.0, 250.0, 750.0])
    np.testing.assert_allclose(fsk4.freq_map, expected_freqs)

    # Test 8-FSK
    fsk8 = sdr.FSK(8, freq_sep=200)
    expected_freqs = np.array([-700.0, -500.0, -300.0, -100.0, 100.0, 300.0, 500.0, 700.0])
    np.testing.assert_allclose(fsk8.freq_map, expected_freqs)


def test_modulate_deterministic():
    """Test that modulation is deterministic."""
    fsk = sdr.FSK(4, freq_sep=500, sps=8)
    symbols = np.array([0, 1, 2, 3, 2, 1, 0])

    # Multiple calls should produce identical results
    tx_signal1 = fsk.modulate(symbols)
    tx_signal2 = fsk.modulate(symbols)

    np.testing.assert_allclose(tx_signal1, tx_signal2)
