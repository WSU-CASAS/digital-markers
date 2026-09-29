from dataclasses import dataclass

@dataclass(frozen=True)
class MarkerRequirements:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
