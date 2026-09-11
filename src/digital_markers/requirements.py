from dataclasses import dataclass

@dataclass(frozen=True)
class MarkerRequirements:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
