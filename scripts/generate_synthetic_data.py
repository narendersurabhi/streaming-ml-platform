from streaming_ml_platform.data.synthetic import generate_synthetic_data
from streaming_ml_platform.paths import RAW_DIR
from streaming_ml_platform.utils.io import write_csv


if __name__ == "__main__":
    frames = generate_synthetic_data()
    for name, df in frames.items():
        write_csv(df, RAW_DIR / f"{name}.csv")
    print("Synthetic data generated")
