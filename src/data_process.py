import pandas as pd
import numpy as np
import copy


def extract_track_info(result) -> pd.DataFrame:
    """Extract tracking information from YOLO detection results.

    Processes YOLO detection results to extract class IDs, names, confidence scores,
    track IDs, and bounding boxes for all detected objects. Adds a mask column
    initialized with NaN values for later filtering.

    Args:
        result: YOLO detection result object containing boxes and metadata.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - class_id (int): Object class identifier
            - class_name (str): Human-readable class name
            - confidence (float): Detection confidence score
            - track_id (int or None): Unique tracking identifier
            - bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            - mask (float): Filtering mask, initialized as NaN
    """
    trackin_data = []
    if result.boxes is not None:
        for box in result.boxes:
            data = {
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "track_id": (
                    int(box.id[0])
                    if hasattr(box, "id") and box.id is not None
                    else None
                ),
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
            }
            trackin_data.append(data)

    df = pd.DataFrame(trackin_data)
    # Add mask column to current_df NaN
    df["mask"] = np.nan
    return df


def update_all_tracks(all_tracks_df, current_df):
    """Update the master tracking DataFrame with current frame detections.

    Merges current frame tracking data with the accumulated tracking history.
    New track IDs are added as new rows, while existing track IDs have their
    detection data updated while preserving their mask values.

    Args:
        all_tracks_df (pd.DataFrame): Master DataFrame containing all historical
            tracking data with preserved mask values.
        current_df (pd.DataFrame): Current frame tracking data from extract_track_info().

    Returns:
        pd.DataFrame: Updated master DataFrame with:
            - New tracks appended with current frame data
            - Existing tracks updated with latest detection info
            - Mask values preserved for existing tracks
    """
    if current_df.empty:
        return all_tracks_df

    # Get existing track_ids
    existing_track_ids = set(all_tracks_df["track_id"].dropna())
    current_track_ids = set(current_df["track_id"].dropna())

    # Find new track_ids that need to be added
    new_track_ids = current_track_ids - existing_track_ids

    # Find existing track_ids that need to be updated
    update_track_ids = current_track_ids & existing_track_ids

    # Add new tracks
    if new_track_ids:
        new_tracks = current_df[current_df["track_id"].isin(new_track_ids)]
        all_tracks_df = pd.concat([all_tracks_df, new_tracks], ignore_index=True)

    # Update existing tracks
    if update_track_ids:
        for track_id in update_track_ids:
            # Get the row index in all_tracks_df for this track_id
            row_idx = all_tracks_df[all_tracks_df["track_id"] == track_id][
                "track_id"
            ].index[0]

            # Get the updated data from current_df
            # the .iloc[0] ensures we get a Series instead of a DataFrame
            updated_data = current_df[current_df["track_id"] == track_id].iloc[0]

            # Update all columns except mask (preserve existing mask value)
            existing_mask = all_tracks_df.loc[row_idx]["mask"]
            # Update columns individually to avoid array assignment issues
            all_tracks_df.loc[row_idx, "class_id"] = updated_data["class_id"]
            all_tracks_df.loc[row_idx, "class_name"] = updated_data["class_name"]
            all_tracks_df.loc[row_idx, "confidence"] = updated_data["confidence"]
            # Use .at for single value assignment to avoid potential issues
            all_tracks_df.at[row_idx, "bbox"] = list(updated_data["bbox"])

            # Keep existing mask value
            all_tracks_df.loc[row_idx, "mask"] = existing_mask

    return all_tracks_df


def filter_results_by_mask(results, all_tracks_df):
    """Filter YOLO detection results based on LLM categorization masks.

    Creates a filtered copy of YOLO results containing only detections whose
    track IDs have mask value of 1 (indicating they match the user query).
    This enables selective visualization of only relevant objects.

    Args:
        results (list): Original YOLO detection results from model.track().
        all_tracks_df (pd.DataFrame): Master tracking DataFrame containing
            mask values from LLM categorization.

    Returns:
        YOLO Results object: Deep copy of original results with boxes filtered
            to include only tracks where mask == 1. Can be used directly with
            .plot() method for annotated visualization.

    Note:
        Returns a deep copy to avoid modifying the original YOLO results.
        If no boxes match the filter criteria, returns results with empty boxes.
    """
    # Get list of track_ids where mask == 1
    active_ids = set(
        all_tracks_df.loc[all_tracks_df["mask"] == 1, "track_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    # Make a deep copy so you don’t overwrite the original YOLO result
    filtered_results = copy.deepcopy(results[0])

    if filtered_results.boxes is None:
        return filtered_results  # no detections anyway

    # Keep only boxes whose track_id is in active_ids
    keep_indices = []
    for i, box in enumerate(filtered_results.boxes):
        if hasattr(box, "id") and box.id is not None:
            if int(box.id.item()) in active_ids:
                keep_indices.append(i)

    # Subset the boxes tensor
    filtered_results.boxes = filtered_results.boxes[keep_indices]

    return filtered_results
