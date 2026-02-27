import arxiv
import pandas as pd
from datetime import datetime

class BenchmarkTracker:
    def __init__(self):
        """
        Initialize the BenchmarkTracker with an empty DataFrame.
        """
        self.data = pd.DataFrame(columns=['Model', 'Date', 'Score', 'Benchmark Name'])

    def add_benchmark(self, model, date, score, benchmark_name):
        """
        Add a single benchmark data point.

        Args:
            model (str): Name of the model.
            date (str): Date of the benchmark in 'YYYY-MM-DD' format.
            score (float): The benchmark score.
            benchmark_name (str): The name of the benchmark (e.g., 'MMLU').
        """
        new_row = {
            'Model': model,
            'Date': pd.to_datetime(date),
            'Score': score,
            'Benchmark Name': benchmark_name
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

    def fetch_arxiv_papers(self, query, max_results=10):
        """
        Fetch recent ML papers from arXiv based on a query.

        Args:
            query (str): The search query (e.g., "MMLU", "HumanEval").
            max_results (int): Maximum number of results to return.

        Returns:
            list: A list of dictionaries containing paper details.
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        client = arxiv.Client()
        for result in client.results(search):
            papers.append({
                'Title': result.title,
                'Date': result.published,
                'Summary': result.summary,
                'Authors': [a.name for a in result.authors],
                'Link': result.entry_id
            })
        return papers

    def get_benchmark_data(self, benchmark_name=None):
        """
        Retrieve benchmark data, optionally filtered by benchmark name.

        Args:
            benchmark_name (str, optional): The name of the benchmark to filter by.

        Returns:
            pd.DataFrame: The filtered benchmark data.
        """
        if benchmark_name:
            return self.data[self.data['Benchmark Name'] == benchmark_name]
        return self.data
