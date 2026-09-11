from dataclasses import dataclass

@dataclass
class Demographics:
    age: float | None = None
    race: str | None = None
    ethnicity: str | None = None
    gender: str | None = None
    education: str | None = None
    marital_status: str | None = None
    num_children: int | None = None
    lives_alone: bool | None = None
