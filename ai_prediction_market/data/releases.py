import requests
import pandas as pd
from datetime import datetime

class ReleaseTracker:
    def __init__(self):
        self.releases = pd.DataFrame(columns=['Company', 'Product', 'Date'])

    def add_release(self, company, product, date):
        """
        Add a product release manually.

        Args:
            company (str): Company name (e.g., 'OpenAI').
            product (str): Product name (e.g., 'GPT-4').
            date (str): Release date in 'YYYY-MM-DD'.
        """
        new_row = {
            'Company': company,
            'Product': product,
            'Date': pd.to_datetime(date)
        }
        self.releases = pd.concat([self.releases, pd.DataFrame([new_row])], ignore_index=True)

    def fetch_github_releases(self, repo_owner, repo_name):
        """
        Fetch releases from a GitHub repository.

        Args:
            repo_owner (str): Owner of the repo (e.g., 'huggingface').
            repo_name (str): Name of the repo (e.g., 'transformers').

        Returns:
            list: List of release dictionaries.
        """
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
        response = requests.get(url)

        if response.status_code == 200:
            releases = response.json()
            fetched_data = []
            for release in releases:
                fetched_data.append({
                    'Product': release['tag_name'],
                    'Date': release['published_at'],
                    'Company': repo_owner
                })
            return fetched_data
        else:
            print(f"Failed to fetch releases: {response.status_code}")
            return []

    def get_releases(self, company=None):
        if company:
            return self.releases[self.releases['Company'] == company]
        return self.releases
