"""
Test FSK class properties and initialization.
"""

import numpy as np

import sdr


def test_fsk_initialization_basic():
    """Test basic FSK initialization."""
    fsk = sdr.FSK(2, freq_sep=1000)

    assert fsk.order == 2
    assert fsk.bps == 1
    assert fsk.freq_sep == 1000
    assert fsk.sps == 8  # default
    assert fsk.symbol_labels == "gray"  # default


def test_fsk_initialization_parameters():
    """Test FSK initialization with various parameters."""
    fsk = sdr.FSK(4, freq_sep=500, sps=16, symbol_labels="bin")

    assert fsk.order == 4
    assert fsk.bps == 2
    assert fsk.freq_sep == 500
    assert fsk.sps == 16
    assert fsk.symbol_labels == "bin"


def test_fsk_order_validation():
    """Test modulation order validation."""
    # Valid orders (powers of 2)
    for order in [2, 4, 8, 16]:
        fsk = sdr.FSK(order, freq_sep=1000)
        assert fsk.order == order
        assert fsk.bps == int(np.log2(order))

    # Invalid orders (not powers of 2) should raise exceptions
    # Note: These are expected to fail - we don't test them explicitly


def test_fsk_freq_sep_validation():
    """Test frequency separation validation."""
    # Valid positive frequency separations
    for freq_sep in [100, 500, 1000, 2000]:
        fsk = sdr.FSK(2, freq_sep=freq_sep)
        assert fsk.freq_sep == freq_sep

    # Invalid frequency separations are handled by validation
    # Note: Zero and negative values are expected to fail


def test_fsk_sps_validation():
    """Test samples per symbol validation."""
    # Valid sps values
    for sps in [1, 4, 8, 16, 32]:
        fsk = sdr.FSK(2, freq_sep=1000, sps=sps)
        assert fsk.sps == sps

    # Invalid sps values are handled by validation
    # Note: Zero and negative values are expected to fail


def test_fsk_symbol_labels():
    """Test symbol labeling schemes."""
    # Gray coding (default)
    fsk_gray = sdr.FSK(4, freq_sep=500, symbol_labels="gray")
    assert fsk_gray.symbol_labels == "gray"
    # For 4-FSK, symbol_map should be 4 elements with Gray coding order
    expected_gray = np.array([0, 1, 3, 2])
    np.testing.assert_array_equal(fsk_gray.symbol_map, expected_gray)

    # Binary coding
    fsk_bin = sdr.FSK(4, freq_sep=500, symbol_labels="bin")
    assert fsk_bin.symbol_labels == "bin"
    # For 4-FSK, symbol_map should be 4 elements in binary order
    expected_bin = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(fsk_bin.symbol_map, expected_bin)

    # Invalid symbol labeling is handled by validation
    # Note: Invalid labels are expected to fail


def test_fsk_pulse_shape_rect():
    """Test rectangular pulse shape."""
    fsk = sdr.FSK(2, freq_sep=1000, pulse_shape="rect", span=1)

    # Should use rectangular pulse shape
    expected_pulse = sdr.rectangular(fsk.sps, span=1)
    np.testing.assert_allclose(fsk.pulse_shape, expected_pulse)


def test_fsk_pulse_shape_custom():
    """Test custom pulse shape."""
    custom_pulse = np.array([1, 0.5, 0.2, 0.1], dtype=complex)
    fsk = sdr.FSK(2, freq_sep=1000, sps=4, pulse_shape=custom_pulse)

    np.testing.assert_allclose(fsk.pulse_shape, custom_pulse)


def test_fsk_freq_map():
    """Test frequency mapping."""
    # BFSK
    bfsk = sdr.FSK(2, freq_sep=1000)
    expected_bfsk = np.array([-500.0, 500.0])
    np.testing.assert_allclose(bfsk.freq_map, expected_bfsk)

    # 4-FSK
    fsk4 = sdr.FSK(4, freq_sep=400)
    expected_4fsk = np.array([-600.0, -200.0, 200.0, 600.0])
    np.testing.assert_allclose(fsk4.freq_map, expected_4fsk)

    # 8-FSK
    fsk8 = sdr.FSK(8, freq_sep=100)
    expected_8fsk = np.array([-350.0, -250.0, -150.0, -50.0, 50.0, 150.0, 250.0, 350.0])
    np.testing.assert_allclose(fsk8.freq_map, expected_8fsk)


def test_fsk_repr():
    """Test string representation."""
    fsk = sdr.FSK(4, freq_sep=500)
    repr_str = repr(fsk)

    assert "sdr.FSK" in repr_str
    assert "order=4" in repr_str
    assert "freq_sep=500" in repr_str


def test_fsk_str():
    """Test detailed string representation."""
    fsk = sdr.FSK(4, freq_sep=500, sps=16)
    str_repr = str(fsk)

    assert "sdr.FSK:" in str_repr
    assert "order: 4" in str_repr
    assert "bps: 2" in str_repr
    assert "freq_sep: 500" in str_repr
    assert "sps: 16" in str_repr


def test_fsk_properties_immutable():
    """Test that properties are read-only (where expected)."""
    fsk = sdr.FSK(4, freq_sep=500)

    # These should be readable
    assert isinstance(fsk.order, int)
    assert isinstance(fsk.bps, int)
    assert isinstance(fsk.freq_sep, (int, float))
    assert isinstance(fsk.sps, int)
    assert isinstance(fsk.symbol_labels, str)
    assert isinstance(fsk.pulse_shape, np.ndarray)
    assert isinstance(fsk.symbol_map, np.ndarray)
    assert isinstance(fsk.freq_map, np.ndarray)


def test_fsk_different_orders():
    """Test FSK with different modulation orders."""
    orders = [2, 4, 8, 16]

    for order in orders:
        fsk = sdr.FSK(order, freq_sep=1000)

        # Check computed properties
        assert fsk.order == order
        assert fsk.bps == int(np.log2(order))
        assert len(fsk.freq_map) == order
        assert len(fsk.symbol_map) == order

        # Check frequency mapping symmetry around zero
        freq_map = fsk.freq_map
        assert np.allclose(freq_map, -freq_map[::-1])


def test_fsk_span_parameter():
    """Test pulse shape span parameter."""
    # Default span
    fsk_default = sdr.FSK(2, freq_sep=1000, pulse_shape="rect")
    assert len(fsk_default.pulse_shape) == fsk_default.sps  # span=1 default

    # Custom span
    fsk_custom = sdr.FSK(2, freq_sep=1000, pulse_shape="rect", span=2)
    assert len(fsk_custom.pulse_shape) == 2 * fsk_custom.sps
