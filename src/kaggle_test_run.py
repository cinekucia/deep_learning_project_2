import argparse
import os
import pandas as pd
from trainer.test import test
from lightning.pytorch.loggers import CSVLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Kaggle test.")
    parser.add_argument("model_name", type=str, help="Name of the model to test.")

    args = parser.parse_args()

    # Prepare the configuration dictionary
    config_dict = {"model_name": args.model_name}

    # Initialize the CSV logger
    csv_logger = CSVLogger("logs", name=f"test_{args.model_name}")

    # Example directory structure, adjust as necessary
    audio_dir = 'C:/Users/Filip/Desktop/PW/2 semestr/dl_test/DL_2_transformers_Plichta_Kucia_Taczala/data'  # This needs to be specified or determined
    model_path = os.path.join('/path/to/models', f"{args.model_name}.pth")

    file_names, labels = test(config_dict, audio_dir, csv_logger, model_path)

    df = pd.DataFrame({
        "fname": file_names,
        "label": labels,
    })

    # Save the dataframe to a CSV file
    results_path = f"results/kaggle_results_{args.model_name}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, sep=",", index=False)

    print(f"Results saved to {results_path}")
