"""Walking-bout detection from detected step times."""

import numpy as np


MAX_STEP_GAP_SECONDS = 2.0
MIN_STEPS_PER_BOUT = 5


def count_walking_bouts(
	step_times: np.ndarray,
	*,
	max_gap_seconds: float = MAX_STEP_GAP_SECONDS,
	min_steps: int = MIN_STEPS_PER_BOUT,
) -> int:
	if len(step_times) < min_steps:
		return 0

	bout_count = 0
	current_steps = 1
	for gap in np.diff(step_times):
		if gap <= max_gap_seconds:
			current_steps += 1
		else:
			if current_steps >= min_steps:
				bout_count += 1
			current_steps = 1

	if current_steps >= min_steps:
		bout_count += 1
	return int(bout_count)


__all__ = ["MAX_STEP_GAP_SECONDS", "MIN_STEPS_PER_BOUT", "count_walking_bouts"]
