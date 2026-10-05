"""Pipeline runner for LLM-as-Judge news labeling (Step 09)."""
import argparse
from pipeline.llm_labeler import run_labeling_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge Financial News Labeler")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of articles to label")
    parser.add_argument("--model", type=str, default="mock-judge", help="Model name (e.g. gpt-4o-mini, mock-judge)")
    args = parser.parse_args()

    run_labeling_pipeline(sample_size=args.sample_size, model=args.model)
