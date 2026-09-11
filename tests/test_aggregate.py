import pandas as pd
from digital_markers.statistics import summarize_series

def test_summarize_series():
    result = summarize_series(pd.Series([1.0, 2.0, 3.0, 4.0]), prefix="example")
    assert result["mean_example"] == 2.5
