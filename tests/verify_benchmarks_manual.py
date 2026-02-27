from ai_prediction_market.data.benchmarks import BenchmarkTracker

def main():
    print("Initializing BenchmarkTracker...")
    tracker = BenchmarkTracker()

    print("Adding sample data...")
    tracker.add_benchmark("GPT-4", "2023-03-14", 86.4, "MMLU")
    tracker.add_benchmark("Claude 3 Opus", "2024-03-04", 86.8, "MMLU")

    print("\nBenchmark Data:")
    print(tracker.get_benchmark_data())

    print("\nFetching arXiv papers for 'MMLU'...")
    try:
        papers = tracker.fetch_arxiv_papers("MMLU", max_results=2)
        for i, paper in enumerate(papers):
            print(f"Paper {i+1}: {paper['Title']} ({paper['Date']})")
    except Exception as e:
        print(f"Error fetching arXiv papers: {e}")

if __name__ == "__main__":
    main()
