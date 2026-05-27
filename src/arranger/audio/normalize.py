"""
Pure audio normalization and gain utilities.

These are stateless helper functions with no dependencies on pydub or other audio libraries.
"""

import math


def db_to_linear(db: float) -> float:
    """
    Convert decibels to linear gain.

    Args:
        db: Gain in decibels (e.g., -6.0 for half amplitude)

    Returns:
        Linear gain multiplier (e.g., 0.5 for -6 dB)

    Examples:
        >>> db_to_linear(0.0)
        1.0
        >>> db_to_linear(-6.0)  # approximately 0.5
        0.5011872336272722
        >>> db_to_linear(-20.0)  # 0.1
        0.1
    """
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """
    Convert linear gain to decibels.

    Args:
        linear: Linear gain multiplier (must be > 0)

    Returns:
        Gain in decibels

    Examples:
        >>> linear_to_db(1.0)
        0.0
        >>> linear_to_db(0.5)  # approximately -6 dB
        -6.020599913279624
        >>> linear_to_db(0.1)  # -20 dB
        -20.0
    """
    if linear <= 0:
        return float('-inf')
    return 20.0 * math.log10(linear)


def apply_gain_db(gain_db: float) -> float:
    """
    Get linear gain multiplier from dB, with optimization for 0 dB.

    This is the common pattern used throughout the codebase:
        gain = 10 ** (gain_db / 20.0) if gain_db != 0.0 else 1.0

    Args:
        gain_db: Gain adjustment in decibels

    Returns:
        Linear gain multiplier (1.0 if gain_db is 0)
    """
    if gain_db == 0.0:
        return 1.0
    return db_to_linear(gain_db)
