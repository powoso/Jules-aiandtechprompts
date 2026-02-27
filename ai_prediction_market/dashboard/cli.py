import argparse
import pandas as pd
from ai_prediction_market.data.benchmarks import BenchmarkTracker
from ai_prediction_market.data.releases import ReleaseTracker
from ai_prediction_market.analysis.modeling import BenchmarkModel, CadenceModel
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="AI Prediction Market CLI")
    parser.add_argument("--fetch-arxiv", action="store_true", help="Fetch recent ML papers from arXiv")
    parser.add_argument("--fetch-github", action="store_true", help="Fetch GitHub releases")
    args = parser.parse_args()

    print("=== AI Prediction Market Dashboard ===")

    # Initialize Trackers
    benchmark_tracker = BenchmarkTracker()
    release_tracker = ReleaseTracker()

    # Load Sample Data (Simulated)
    print("\n[Loading Sample Data...]")
    benchmark_tracker.add_benchmark("GPT-4", "2023-03-14", 86.4, "MMLU")
    benchmark_tracker.add_benchmark("Claude 3 Opus", "2024-03-04", 86.8, "MMLU")
    benchmark_tracker.add_benchmark("Llama 3", "2024-04-18", 82.0, "MMLU")

    release_tracker.add_release("OpenAI", "GPT-3", "2020-06-11")
    release_tracker.add_release("OpenAI", "GPT-3.5", "2022-11-30")
    release_tracker.add_release("OpenAI", "GPT-4", "2023-03-14")
    release_tracker.add_release("OpenAI", "GPT-4o", "2024-05-13")

    # Fetch Real Data if requested
    if args.fetch_arxiv:
        print("\n[Fetching arXiv Data...]")
        papers = benchmark_tracker.fetch_arxiv_papers("MMLU", max_results=3)
        for p in papers:
            print(f"- {p['Title']} ({p['Date']})")

    if args.fetch_github:
         print("\n[Fetching GitHub Releases...]")
         releases = release_tracker.fetch_github_releases("huggingface", "transformers")
         print(f"Fetched {len(releases)} releases from huggingface/transformers.")

    # Run Models
    print("\n[Running Analysis...]")

    # 1. Benchmark Prediction
    benchmark_model = BenchmarkModel()
    mmlu_data = benchmark_tracker.get_benchmark_data("MMLU")
    if not mmlu_data.empty and len(mmlu_data) > 1:
        benchmark_model.fit(mmlu_data)
        future_date = (datetime.now() + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        predicted_score = benchmark_model.predict(future_date)
        print(f"-> Projected MMLU Score on {future_date}: {predicted_score:.2f}")
    else:
        print("-> Not enough MMLU data for prediction.")

    # 2. Release Cadence Prediction
    cadence_model = CadenceModel()
    openai_releases = release_tracker.get_releases("OpenAI")
    if not openai_releases.empty and len(openai_releases) > 1:
        next_release = cadence_model.predict_next_release(openai_releases)
        if next_release:
             print(f"-> Predicted Next OpenAI Release Date: {next_release.date()}")
    else:
        print("-> Not enough OpenAI release data for prediction.")

    print("\n=== End Report ===")

if __name__ == "__main__":
    main()
