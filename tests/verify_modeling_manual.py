from ai_prediction_market.analysis.modeling import BenchmarkModel, CadenceModel
import pandas as pd

def main():
    # Test BenchmarkModel
    print("Testing BenchmarkModel...")
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'Score': [10.0, 11.0, 12.0]
    })

    model = BenchmarkModel()
    model.fit(data)

    future_date = '2023-04-01'
    prediction = model.predict(future_date)
    print(f"Predicted score for {future_date}: {prediction:.2f}")

    # Test CadenceModel
    print("\nTesting CadenceModel...")
    release_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-02-01', '2023-03-01']
    })

    cadence_model = CadenceModel()
    next_release = cadence_model.predict_next_release(release_data)
    print(f"Predicted next release date: {next_release.date()}")

if __name__ == "__main__":
    main()
