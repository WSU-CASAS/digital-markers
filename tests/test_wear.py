import numpy as np
import pandas as pd

from digital_markers.preprocessing import compute_wear_prob, detect_nonwear


def test_detect_nonwear_uses_acceleration_and_rotation_variability():
    count = 120
    timestamps = pd.date_range("2026-01-01", periods=count, freq="100ms")
    active_acceleration = np.tile([0.0, 0.1], 30)
    active_rotation = np.tile([0.0, 0.2], 30)
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": np.r_[active_acceleration, np.zeros(60)],
            "acceleration_y": np.zeros(count),
            "acceleration_z": np.zeros(count),
            "rotation_rate_x": np.r_[active_rotation, np.zeros(60)],
            "rotation_rate_y": np.zeros(count),
            "rotation_rate_z": np.zeros(count),
        },
        index=np.arange(100, 100 + count),
    )

    worn = detect_nonwear(motion, rate=1)
    probability = compute_wear_prob(motion, rate=1)

    assert worn.index.equals(motion.index)
    assert worn.iloc[:60].all()
    assert not worn.iloc[60:].any()
    assert probability.iloc[:60].eq(1.0).all()
    assert probability.iloc[60:].eq(0.0).all()


def test_detect_nonwear_requires_acceleration_and_rotation_columns():
    motion = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="100ms"),
            "acceleration_x": [0.0, 0.1],
            "acceleration_y": [0.0, 0.0],
            "acceleration_z": [0.0, 0.0],
        }
    )

    try:
        detect_nonwear(motion)
    except ValueError as exc:
        assert "rotation_rate_x, rotation_rate_y, rotation_rate_z" in str(exc)
    else:
        raise AssertionError("Expected missing rotation axes to raise ValueError")