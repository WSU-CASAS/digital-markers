from dataclasses import dataclass, field
from digital_markers.data.participant import ParticipantData

@dataclass
class MarkerContext:
    data: ParticipantData
    derived: dict[str, object] = field(default_factory=dict)

    def get_derived(self, name: str) -> object | None:
        return self.derived.get(name)

    def set_derived(self, name: str, value: object) -> None:
        self.derived[name] = value
