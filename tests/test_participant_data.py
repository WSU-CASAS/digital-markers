import pandas as pd
from digital_markers import ParticipantData

def test_provided_activities_are_input_data():
    activities = pd.DataFrame({"start_time": ["2026-09-01T08:00:00Z"], "end_time": ["2026-09-01T08:30:00Z"], "activity": ["eat"], "source": ["participant"]})
    data = ParticipantData(participant_id="P001", activities=activities)
    assert data.activities.iloc[0]["source"] == "participant"

def test_tasks_are_keyed_by_name():
    n_back = pd.DataFrame({"trial": [1], "correct": [True]})
    data = ParticipantData(participant_id="P001", tasks={"n_back": n_back})
    assert data.tasks["n_back"].iloc[0]["correct"]
