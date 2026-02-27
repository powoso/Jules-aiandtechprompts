# AI Prediction Market System

This repository implements a system for tracking AI benchmarks and product release cadences, designed to assist in AI/tech prediction markets. It leverages data from arXiv and GitHub to model trends and predict future outcomes.

## Features

- **Benchmark Tracking:** Fetches and stores AI benchmark scores (e.g., MMLU, HumanEval). Integrates with the arXiv API to track relevant research papers.
- **Release Cadence Modeling:** Tracks product release dates (e.g., from GitHub) and predicts future release windows based on historical cadence.
- **Prediction Dashboard:** CLI tool to view current data and run predictive models.

## Installation on macOS

Follow these steps to set up the project on macOS:

1.  **Prerequisites:** Ensure you have Python 3.8 or higher installed. You can check this by running:
    ```bash
    python3 --version
    ```
    If not installed, download it from [python.org](https://www.python.org/) or install via Homebrew:
    ```bash
    brew install python
    ```

2.  **Clone the Repository:**
    Open Terminal and clone the project:
    ```bash
    git clone <repository_url>
    cd ai-prediction-market
    ```

3.  **Create a Virtual Environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the prediction dashboard CLI:

```bash
python3 ai_prediction_market/dashboard/cli.py
```

Options:
- `--fetch-arxiv`: Fetch recent ML papers from arXiv.
- `--fetch-github`: Fetch recent releases from GitHub (e.g., huggingface/transformers).

Example:
```bash
python3 ai_prediction_market/dashboard/cli.py --fetch-arxiv
```

## Running Tests

To run the unit tests:

```bash
pytest
```
