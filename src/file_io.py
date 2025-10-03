from datetime import datetime
import pandas as pd


def create_csv_file():
    """Create a new timestamped CSV file for tracking data logging.

    Generates a unique CSV file with current timestamp in the filename to store
    drone tracking experiment data. The file is initialized with an empty DataFrame
    containing the proper column structure and data types for tracking information.

    Returns:
        str: Full file path to the created CSV file in format:
            'data/logs/tracking_logs_YYYYMMDD_HHMMSS.csv'

    Example:
        >>> csv_file = create_csv_file()
        >>> print(csv_file)
        'data/logs/tracking_logs_20241003_143052.csv'

    Note:
        - Creates 'data/logs/' directory if it doesn't exist
        - Each call generates a unique filename based on current timestamp
        - File is initialized with proper column structure for tracking data:
          track_id, class_id, class_name, confidence, bbox, mask
    """
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
