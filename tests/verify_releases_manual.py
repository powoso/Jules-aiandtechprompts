from ai_prediction_market.data.releases import ReleaseTracker

def main():
    print("Initializing ReleaseTracker...")
    tracker = ReleaseTracker()

    print("Adding sample data...")
    tracker.add_release("OpenAI", "GPT-4", "2023-03-14")
    tracker.add_release("Anthropic", "Claude 3", "2024-03-04")

    print("\nRelease Data:")
    print(tracker.get_releases())

    print("\nFetching GitHub releases for 'huggingface/transformers'...")
    try:
        releases = tracker.fetch_github_releases("huggingface", "transformers")
        if releases:
            for i, release in enumerate(releases[:2]):
                 print(f"Release {i+1}: {release['Product']} ({release['Date']})")
        else:
            print("No releases found or failed to fetch.")
    except Exception as e:
        print(f"Error fetching GitHub releases: {e}")

if __name__ == "__main__":
    main()
