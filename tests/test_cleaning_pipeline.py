from cleaning.pipeline import run_cleaning_pipeline


def test_pipeline():

    file_path = "uploaded_files/churn/data.csv"
    target="CustomerChurned"
    result = run_cleaning_pipeline(file_path, target)

    print("Dataset ID:", result["dataset_id"])
    print("Cleaned file:", result["cleaned_file_path"])
    print("Metadata:", result["metadata"])
    print("Post clean report:", result["post_clean_report"])

if __name__ == "__main__":
    test_pipeline()