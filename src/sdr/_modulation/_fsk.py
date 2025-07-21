"""
A module containing classes for frequency-shift keying (FSK) modulations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.special
from typing_extensions import Literal

from .._conversion import ebn0_to_esn0, linear
from .._helper import (
    convert_output,
    export,
    verify_arraylike,
    verify_bool,
    verify_literal,
    verify_scalar,
)
from ._pulse_shapes import rectangular


@export
class FSK:
    r"""
    Implements frequency-shift keying (FSK) modulation and demodulation.

    Notes:
        Frequency-shift keying (FSK) is a non-linear frequency modulation scheme that encodes information by
        modulating the frequency of a carrier sinusoid. The modulation order $M = 2^k$ is a power of 2 and
        indicates the number of frequencies used. The input bit stream is taken $k$ bits at a time to create
        decimal symbols $s[k] \in \{0, \dots, M-1\}$. These decimal symbols $s[k]$ are then mapped to
        frequencies $f[k]$ by the equation

        $$f[k] = f_c + \left(s[k] - \frac{M-1}{2}\right) \cdot \Delta f$$

        where $f_c$ is the center frequency and $\Delta f$ is the frequency separation.

        The complex baseband signal is generated as:

        $$x[n] = \exp\left(j 2\pi \frac{f[k] - f_c}{f_s} n\right) \cdot h[n]$$

        where $h[n]$ is the pulse shape and $f_s$ is the sampling rate.

    .. nomenclature::
        :collapsible:

        - $k$: Symbol index
        - $n$: Sample index
        - $s[k]$: Decimal symbols
        - $f[k]$: Frequency symbols (Hz)
        - $x[n]$: Pulse-shaped complex samples
        - $\tilde{x}[n]$: Received (noisy) pulse-shaped complex samples
        - $\hat{s}[k]$: Decimal symbol decisions
        - $M$: Modulation order
        - $f_c$: Center frequency (Hz)
        - $\Delta f$: Frequency separation (Hz)
        - $f_s$: Sampling rate (Hz)

    Examples:
        Create a binary FSK (BFSK) modem with 1 kHz frequency separation.

        .. ipython:: python

            bfsk = sdr.FSK(2, freq_sep=1000, sps=8); bfsk

        Create a 4-FSK modem with 500 Hz frequency separation.

        .. ipython:: python

            fsk4 = sdr.FSK(4, freq_sep=500, sps=8); fsk4

    Group:
        modulation-orthogonal
    """

    def __init__(
        self,
        order: int,
        freq_sep: float,
        sps: int = 8,
        pulse_shape: npt.ArrayLike | Literal["rect"] = "rect",
        span: int | None = None,
        symbol_labels: Literal["bin", "gray"] = "gray",
    ):
        r"""
        Creates a new FSK modulation object.

        Arguments:
            order: The modulation order $M$. Must be a power of 2.
            freq_sep: The frequency separation $\Delta f$ in Hz between adjacent symbols.
            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape.
                - `"rect"`: Rectangular pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.
                If `None`, 1 is used for `"rect"`.
            symbol_labels: The symbol labeling scheme.

                - `"bin"`: Binary labeling scheme where the symbols are $\\{0, 1, 2, \\ldots, M-1\\}$.
                - `"gray"`: Gray labeling scheme where adjacent symbols only differ by one bit.

        See Also:
            sdr.rectangular
        """
        self._order = verify_scalar(order, int=True, positive=True, power_of_two=True)  # Modulation order
        self._bps = int(np.log2(self._order))  # Coded bits per symbol
        self._freq_sep = verify_scalar(freq_sep, float=True, positive=True)  # Frequency separation
        self._sps = verify_scalar(sps, int=True, positive=True)  # Samples per symbol
        verify_literal(symbol_labels, ["bin", "gray"])
        self._symbol_labels = symbol_labels

        if isinstance(pulse_shape, str):
            verify_literal(pulse_shape, ["rect"])
            if pulse_shape == "rect":
                if span is None:
                    span = 1
                self._pulse_shape = rectangular(self.sps, span=span)
        else:
            self._pulse_shape = verify_arraylike(pulse_shape, complex=True, ndim=1)

        # Create symbol mapping for bit-to-symbol conversion
        if symbol_labels == "gray":
            if self.bps == 1:
                # Special case for binary (BFSK)
                self._symbol_map = np.array([0, 1])
            elif self.bps == 2:
                # 4-FSK Gray coding: [0, 1, 3, 2]
                self._symbol_map = np.array([0, 1, 3, 2])
            elif self.bps == 3:
                # 8-FSK Gray coding: [0, 1, 3, 2, 6, 7, 5, 4]
                self._symbol_map = np.array([0, 1, 3, 2, 6, 7, 5, 4])
            elif self.bps == 4:
                # 16-FSK Gray coding
                self._symbol_map = np.array([0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8])
            else:
                # Fallback to binary for higher orders
                self._symbol_map = np.arange(self.order)
        else:
            # Binary coding: symbols map directly to decimal
            self._symbol_map = np.arange(self.order)

        # Frequency mapping: symbols map to frequencies relative to center
        # Symbol 0 -> -((M-1)/2) * freq_sep, Symbol M-1 -> +((M-1)/2) * freq_sep
        symbol_indices = np.arange(self.order)
        self._freq_map = (symbol_indices - (self.order - 1) / 2) * self.freq_sep

    def __repr__(self) -> str:
        return f"sdr.FSK(order={self.order}, freq_sep={self.freq_sep})"

    def __str__(self) -> str:
        string = "sdr.FSK:"
        string += f"\n  order: {self.order}"
        string += f"\n  bps: {self.bps}"
        string += f"\n  freq_sep: {self.freq_sep} Hz"
        string += f"\n  sps: {self.sps}"
        string += f"\n  pulse_shape: {self.pulse_shape.shape} shape"
        string += f"\n  symbol_labels: {self.symbol_labels}"
        return string

    @property
    def order(self) -> int:
        """
        The modulation order $M$.
        """
        return self._order

    @property
    def bps(self) -> int:
        r"""
        The number of coded bits per symbol $k = \log_2 M$.
        """
        return self._bps

    @property
    def freq_sep(self) -> float:
        r"""
        The frequency separation $\Delta f$ in Hz.
        """
        return self._freq_sep

    @property
    def sps(self) -> int:
        """
        The number of samples per symbol.
        """
        return self._sps

    @property
    def pulse_shape(self) -> npt.NDArray[np.complex128]:
        """
        The pulse shape $h[n]$.
        """
        return self._pulse_shape

    @property
    def symbol_labels(self) -> str:
        """
        The symbol labeling scheme.
        """
        return self._symbol_labels

    @property
    def symbol_map(self) -> npt.NDArray[np.int_]:
        """
        The symbol mapping from decimal symbols to binary labels.
        """
        return self._symbol_map

    @property
    def freq_map(self) -> npt.NDArray[np.float64]:
        """
        The frequency mapping from decimal symbols to frequency offsets (Hz).
        """
        return self._freq_map

    def map_symbols(self, s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r"""
        Converts decimal symbols to complex baseband symbols using FSK modulation.

        Arguments:
            s: The decimal symbols $s[k]$ to map, $0$ to $M-1$.

        Returns:
            The complex symbols representing the frequency tones.

        Examples:
            Create a 4-FSK modulator and map some symbols.

            .. ipython:: python

                fsk = sdr.FSK(4, freq_sep=100, sps=4)
                symbols = np.array([0, 1, 2, 3])
                complex_symbols = fsk.map_symbols(symbols)
                complex_symbols.shape
        """
        s = verify_arraylike(s, int=True)
        if not np.all((0 <= s) & (s < self.order)):
            raise ValueError(f"All symbols must be in [0, {self.order})")

        # Map symbols to frequency offsets
        freq_offsets = self.freq_map[s]

        # Generate complex exponentials for each frequency
        # For baseband representation, we use normalized frequency (cycles per sample)
        n_samples = len(s) * self.sps
        t = np.arange(n_samples) / self.sps  # Time in symbol periods

        # Repeat each frequency offset for sps samples
        freq_offsets_expanded = np.repeat(freq_offsets, self.sps)

        # Generate the complex exponential (baseband representation)
        complex_symbols = np.exp(1j * 2 * np.pi * freq_offsets_expanded * t / self.sps)

        return convert_output(complex_symbols)

    def modulate(self, s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r"""
        Modulates the decimal symbols into a complex baseband signal.

        Arguments:
            s: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.

        Returns:
            The modulated complex baseband signal $x[n]$.

        Examples:
            Create a BFSK modulator and modulate some symbols.

            .. ipython:: python

                bfsk = sdr.FSK(2, freq_sep=1000, sps=8)
                symbols = np.array([0, 1, 1, 0])
                tx_signal = bfsk.modulate(symbols)
                tx_signal.shape
        """
        s = verify_arraylike(s, int=True)
        if not np.all((0 <= s) & (s < self.order)):
            raise ValueError(f"All symbols must be in [0, {self.order})")

        # Map symbols to frequency offsets
        freq_offsets = self.freq_map[s]

        # Generate the modulated signal
        n_samples = len(s) * self.sps
        x = np.zeros(n_samples, dtype=complex)

        for i, freq_offset in enumerate(freq_offsets):
            start_idx = i * self.sps
            end_idx = start_idx + self.sps

            # Time vector for this symbol
            t = np.arange(self.sps)

            # Generate complex exponential with frequency offset
            # For FSK baseband, use normalized frequency offset
            norm_freq = freq_offset / self.sps  # Normalize by sampling rate (sps)
            phase = 2 * np.pi * norm_freq * t / self.sps
            symbol_signal = np.exp(1j * phase)

            # Apply rectangular pulse shaping (multiply by pulse shape)
            pulse_len = min(len(self.pulse_shape), self.sps)
            symbol_signal[:pulse_len] *= self.pulse_shape[:pulse_len]

            x[start_idx:end_idx] = symbol_signal

        return convert_output(x)

    def demodulate(self, x: npt.ArrayLike) -> npt.NDArray[np.int_]:
        r"""
        Demodulates the complex baseband signal into decimal symbols using non-coherent detection.

        Arguments:
            x: The complex baseband signal $x[n]$ to demodulate.

        Returns:
            The demodulated decimal symbols $\hat{s}[k]$.

        Examples:
            Create a BFSK modem and test demodulation.

            .. ipython:: python

                bfsk = sdr.FSK(2, freq_sep=1000, sps=8)
                symbols = np.array([0, 1, 1, 0])
                tx_signal = bfsk.modulate(symbols)
                rx_symbols = bfsk.demodulate(tx_signal)
                np.array_equal(symbols, rx_symbols)
        """
        x = verify_arraylike(x, complex=True)
        n_symbols = len(x) // self.sps

        # Non-coherent demodulation using energy detection
        decisions = np.zeros(n_symbols, dtype=int)

        for i in range(n_symbols):
            start_idx = i * self.sps
            end_idx = start_idx + self.sps
            symbol_signal = x[start_idx:end_idx]

            # Calculate energy for each possible frequency
            energies = np.zeros(self.order)
            for j, freq_offset in enumerate(self.freq_map):
                # Generate reference signal for this frequency
                t = np.arange(self.sps)
                norm_freq = freq_offset / self.sps
                phase = 2 * np.pi * norm_freq * t / self.sps
                reference = np.exp(1j * phase)

                # Calculate correlation energy
                correlation = np.sum(symbol_signal * reference.conj())
                energies[j] = np.abs(correlation) ** 2

            # Choose the frequency with maximum energy
            decisions[i] = np.argmax(energies)

        return convert_output(decisions)

    def ber(
        self,
        ebn0: npt.ArrayLike,
        diff_encoded: bool = False,
    ) -> npt.NDArray[np.floating]:
        r"""
        Computes the theoretical bit error rate (BER) for the given $E_b/N_0$.

        Arguments:
            ebn0: The bit energy to noise power spectral density ratio $E_b/N_0$ in dB.
            diff_encoded: Indicates whether the symbols are differentially encoded.

        Returns:
            The theoretical BER.

        Examples:
            Compute the BER for BFSK.

            .. ipython:: python

                bfsk = sdr.FSK(2, freq_sep=1000)
                ebn0 = np.arange(0, 15)
                ber = bfsk.ber(ebn0)
                ber[0:5]
        """
        ebn0 = verify_arraylike(ebn0, float=True)
        verify_bool(diff_encoded)

        ebn0_linear = linear(ebn0)

        if self.order == 2:
            # Binary FSK (coherent detection)
            ber = 0.5 * np.exp(-0.5 * ebn0_linear)
            if diff_encoded:
                # For differential encoding, BER ≈ 2 * P_e * (1 - P_e) for small P_e
                ber = 2 * ber * (1 - ber)
        else:
            # M-ary FSK (non-coherent detection)
            # Approximate BER for orthogonal M-FSK
            esn0_linear = ebn0_to_esn0(ebn0_linear, self.bps)

            # Symbol error probability for non-coherent M-FSK
            # P_s = sum_{k=1}^{M-1} (-1)^{k+1} * C(M-1, k) * exp(-k*Es/N0 / (k+1))
            symbol_error_prob = np.zeros_like(esn0_linear)
            for k in range(1, self.order):
                coeff = (-1) ** (k + 1) * scipy.special.comb(self.order - 1, k, exact=True)
                symbol_error_prob += coeff * np.exp(-k * esn0_linear / (k + 1))

            # Convert symbol error to bit error (Gray coding assumption)
            ber = symbol_error_prob * self.order / (2 * (self.order - 1))

            if diff_encoded:
                # Apply differential encoding penalty
                ber = 2 * ber * (1 - ber)

        return convert_output(ber)

    def ser(
        self,
        esn0: npt.ArrayLike,
        diff_encoded: bool = False,
    ) -> npt.NDArray[np.floating]:
        r"""
        Computes the theoretical symbol error rate (SER) for the given $E_s/N_0$.

        Arguments:
            esn0: The symbol energy to noise power spectral density ratio $E_s/N_0$ in dB.
            diff_encoded: Indicates whether the symbols are differentially encoded.

        Returns:
            The theoretical SER.

        Examples:
            Compute the SER for 4-FSK.

            .. ipython:: python

                fsk4 = sdr.FSK(4, freq_sep=500)
                esn0 = np.arange(0, 20)
                ser = fsk4.ser(esn0)
                ser[0:5]
        """
        esn0 = verify_arraylike(esn0, float=True)
        verify_bool(diff_encoded)

        esn0_linear = linear(esn0)

        if self.order == 2:
            # Binary FSK
            ser = 0.5 * np.exp(-0.5 * esn0_linear)
        else:
            # M-ary FSK (non-coherent detection)
            # P_s = sum_{k=1}^{M-1} (-1)^{k+1} * C(M-1, k) * exp(-k*Es/N0 / (k+1))
            ser = np.zeros_like(esn0_linear)
            for k in range(1, self.order):
                coeff = (-1) ** (k + 1) * scipy.special.comb(self.order - 1, k, exact=True)
                ser += coeff * np.exp(-k * esn0_linear / (k + 1))

        if diff_encoded:
            # For differential encoding
            ser = 2 * ser * (1 - ser)

        return convert_output(ser)
