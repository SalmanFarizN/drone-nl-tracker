from datetime import datetime
import pandas as pd


def create_csv_file():
    """Create a new CSV file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data/logs/tracking_logs_{timestamp}.csv"

    # Initialize empty CSV file
    all_tracks_df = pd.DataFrame(
        {
            "track_id": pd.Series([], dtype="Int64"),
            "class_id": pd.Series([], dtype="int"),
            "class_name": pd.Series([], dtype="string"),
            "confidence": pd.Series([], dtype="float"),
            "bbox": pd.Series([], dtype="object"),
            "mask": pd.Series([], dtype="Int64"),
        }
    )
    all_tracks_df.to_csv(csv_filename, index=False)
    return csv_filename
