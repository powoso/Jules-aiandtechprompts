import pytest
import pandas as pd
from datetime import datetime, timedelta
from ai_prediction_market.data.benchmarks import BenchmarkTracker
from ai_prediction_market.data.releases import ReleaseTracker
from ai_prediction_market.analysis.modeling import BenchmarkModel, CadenceModel

# Test BenchmarkTracker
def test_benchmark_tracker_add_data():
    tracker = BenchmarkTracker()
    tracker.add_benchmark("TestModel", "2023-01-01", 90.0, "MMLU")
    data = tracker.get_benchmark_data()
    assert len(data) == 1
    assert data.iloc[0]['Model'] == "TestModel"
    assert data.iloc[0]['Score'] == 90.0

def test_benchmark_tracker_filter():
    tracker = BenchmarkTracker()
    tracker.add_benchmark("M1", "2023-01-01", 10, "A")
    tracker.add_benchmark("M2", "2023-01-02", 20, "B")

    data_a = tracker.get_benchmark_data("A")
    assert len(data_a) == 1
    assert data_a.iloc[0]['Benchmark Name'] == "A"

# Test ReleaseTracker
def test_release_tracker_add_data():
    tracker = ReleaseTracker()
    tracker.add_release("Comp", "Prod", "2023-01-01")
    releases = tracker.get_releases()
    assert len(releases) == 1
    assert releases.iloc[0]['Company'] == "Comp"

def test_release_tracker_filter():
    tracker = ReleaseTracker()
    tracker.add_release("C1", "P1", "2023-01-01")
    tracker.add_release("C2", "P2", "2023-01-02")

    releases_c1 = tracker.get_releases("C1")
    assert len(releases_c1) == 1
    assert releases_c1.iloc[0]['Company'] == "C1"

# Test BenchmarkModel
def test_benchmark_model_prediction():
    model = BenchmarkModel()
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02'],
        'Score': [10.0, 11.0]
    })
    model.fit(data)

    # Linear trend: +1 per day. 2023-01-03 should be 12.0
    prediction = model.predict('2023-01-03')
    assert pytest.approx(prediction, 0.1) == 12.0

def test_benchmark_model_untrained_error():
    model = BenchmarkModel()
    with pytest.raises(ValueError):
        model.predict('2023-01-01')

# Test CadenceModel
def test_cadence_model_prediction():
    model = CadenceModel()
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-03']
    })
    # Avg delta is 2 days. Last date is Jan 3. Next should be Jan 5.
    next_release = model.predict_next_release(data)
    expected_date = pd.to_datetime('2023-01-05')
    assert next_release == expected_date

def test_cadence_model_insufficient_data():
    model = CadenceModel()
    data = pd.DataFrame({
        'Date': ['2023-01-01']
    })
    assert model.predict_next_release(data) is None
