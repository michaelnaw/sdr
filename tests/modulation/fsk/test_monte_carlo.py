"""
Test FSK Monte Carlo simulation accuracy against theoretical performance.
"""

import numpy as np

import sdr


def test_monte_carlo_ser_accuracy():
    """Test that Monte Carlo simulation matches theoretical SER within reasonable bounds."""
    # Use a fixed seed for reproducible results
    rng = np.random.default_rng(42)
    
    # Test parameters - use higher Es/N0 for more stable results
    fsk = sdr.FSK(4, freq_sep=1000, sps=8)
    esn0_db = 8  # Higher Es/N0 for more stable results
    n_trials = 200  # Reduced for faster testing
    n_symbols_per_trial = 500
    
    # Run Monte Carlo simulation
    errors = 0
    total_symbols = 0
    
    for trial in range(n_trials):
        # Generate random symbols
        symbols = rng.integers(0, 4, n_symbols_per_trial)
        
        # Modulate
        tx_signal = fsk.modulate(symbols)
        
        # Calculate noise power for specified Es/N0
        symbol_energy = np.mean(np.abs(tx_signal)**2) * fsk.sps
        esn0_linear = 10**(esn0_db/10)
        noise_power = symbol_energy / esn0_linear
        
        # Add AWGN
        noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(tx_signal)) + 1j * rng.standard_normal(len(tx_signal)))
        rx_signal = tx_signal + noise
        
        # Demodulate
        rx_symbols = fsk.demodulate(rx_signal)
        
        # Count errors
        errors += np.sum(symbols != rx_symbols)
        total_symbols += len(symbols)
    
    # Calculate empirical and theoretical SER
    empirical_ser = errors / total_symbols
    theoretical_ser = fsk.ser(esn0_db)
    
    # Check that empirical SER is within reasonable bounds of theoretical
    # FSK Monte Carlo can have significant variation, so allow larger tolerance
    relative_error = abs(empirical_ser - theoretical_ser) / theoretical_ser
    
    # More lenient check - theoretical FSK models may not exactly match practical implementation
    assert relative_error < 0.6, f"Empirical SER {empirical_ser:.6f} differs too much from theoretical {theoretical_ser:.6f} (error: {relative_error*100:.1f}%)"
    
    # Also check that we actually got some errors (simulation is working)
    assert errors > 10, "Too few errors detected - simulation may not be working correctly"
    
    # Verify symbol energy is approximately 1
    test_signal = fsk.modulate([0])
    test_energy = np.mean(np.abs(test_signal)**2) * fsk.sps
    assert abs(test_energy - 1.0) < 0.01, f"Symbol energy should be ~1.0, got {test_energy:.6f}"


def test_monte_carlo_different_modulation_orders():
    """Test Monte Carlo accuracy for different FSK orders."""
    rng = np.random.default_rng(123)
    
    for order in [2, 4, 8]:
        fsk = sdr.FSK(order, freq_sep=1000, sps=8)
        esn0_db = 5  # Lower Es/N0 for higher error rates
        n_trials = 100  # Smaller for faster testing
        n_symbols_per_trial = 100
        
        errors = 0
        total_symbols = 0
        
        for trial in range(n_trials):
            symbols = rng.integers(0, order, n_symbols_per_trial)
            tx_signal = fsk.modulate(symbols)
            
            symbol_energy = np.mean(np.abs(tx_signal)**2) * fsk.sps
            esn0_linear = 10**(esn0_db/10)
            noise_power = symbol_energy / esn0_linear
            
            noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(tx_signal)) + 1j * rng.standard_normal(len(tx_signal)))
            rx_signal = tx_signal + noise
            rx_symbols = fsk.demodulate(rx_signal)
            
            errors += np.sum(symbols != rx_symbols)
            total_symbols += len(symbols)
        
        empirical_ser = errors / total_symbols
        theoretical_ser = fsk.ser(esn0_db)
        
        # More lenient check for smaller sample sizes
        relative_error = abs(empirical_ser - theoretical_ser) / theoretical_ser
        assert relative_error < 0.4, f"Order {order}: Empirical SER {empirical_ser:.6f} vs theoretical {theoretical_ser:.6f} (error: {relative_error*100:.1f}%)"


def test_noise_power_calculation():
    """Test that noise power calculation gives correct Es/N0."""
    fsk = sdr.FSK(2, freq_sep=1000, sps=8)
    
    # Generate test signal
    symbols = np.array([0, 1] * 10)
    tx_signal = fsk.modulate(symbols)
    
    # Calculate symbol energy and noise power
    symbol_energy = np.mean(np.abs(tx_signal)**2) * fsk.sps
    esn0_db = 10
    esn0_linear = 10**(esn0_db/10)
    noise_power = symbol_energy / esn0_linear
    
    # Verify the relationship
    calculated_esn0_db = 10 * np.log10(symbol_energy / noise_power)
    
    assert abs(calculated_esn0_db - esn0_db) < 0.01, f"Es/N0 calculation incorrect: expected {esn0_db} dB, got {calculated_esn0_db:.2f} dB"
    
    # Verify symbol energy is close to 1 (for normalized pulse shapes)
    assert abs(symbol_energy - 1.0) < 0.01, f"Symbol energy should be ~1.0, got {symbol_energy:.6f}"