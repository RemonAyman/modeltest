"""
Simplified entry script that calls centralized training utilities in
`ml_pipeline.py`. This ensures CV-based selection and safe saving.
"""
from ml_pipeline import run_full_training


def main():
    saved, info = run_full_training(cleaned_csv='cleaned_transport_dataset.csv', cv_splits=5)
    print(info)


if __name__ == '__main__':
    main()
