import pandas as pd
import numpy as np
import cv2
import ollama
import copy


# Function to extract track IDs and bboxes of all detected objects
def extract_track_info(result) -> pd.DataFrame:
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
    return df


def update_all_tracks(all_tracks_df, current_df):
    """
    Update all_tracks_df with current frame data:
    - Add new track_ids
    - Update existing track_ids with new information
    """
    if current_df.empty:
        return all_tracks_df

    # Add mask column to current_df NaN
    current_df = current_df.copy()
    current_df["mask"] = np.nan

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
        # print(f"Added {len(new_track_ids)} new tracks: {list(new_track_ids)}")

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

        # print(
        #     f"Updated {len(update_track_ids)} existing tracks: {list(update_track_ids)}"
        # )

    return all_tracks_df


# Function to loop through all uncategorized tracks and ask the LLM to categorize them
def categorize_tracks_with_llm(all_tracks_df, result):
    non_categorized_tracks = all_tracks_df[all_tracks_df["mask"].isna()]

    if non_categorized_tracks.empty:
        print("No uncategorized tracks found.")
        return all_tracks_df

    for _, track in non_categorized_tracks.iterrows():
        track_id = track["track_id"]
        bbox = np.array(track["bbox"]).astype(int)
        x1, y1, x2, y2 = bbox

        orig_img = result[0].orig_img
        img = orig_img[y1:y2, x1:x2]

        # Convert the NumPy array (BGR not the RGB) to bytes
        _, img_encoded = cv2.imencode(".jpg", img)  # Encode as JPEG
        img_bytes = img_encoded.tobytes()  # Convert to bytes

        # Define the system prompt
        system_prompt = {
            "role": "system",
            "content": "You are an intelligent traffic analyst. You will analyze cropped drone images of vehicles from an overhead birds-eye view. Does the image match the description provided by the user? Please provide an answer as an int either 0 (if FALSE) or 1 (TRUE) for the image.",
        }

        # User query
        user_query = {
            "role": "user",
            "content": "I want to track all blue cars",
            "images": [img_bytes],
        }

        # Send the prompts to the model
        res = ollama.chat(model="gemma3:4b", messages=[system_prompt, user_query])

        response_content = res["message"]["content"].strip()
        print(f"Track ID {track_id} - LLM Response: {response_content}")

        # Update the mask based on LLM response
        if response_content == "1":
            mask_value = 1
        elif response_content == "0":
            mask_value = 0
        else:
            print(
                f"Unexpected response for track ID {track_id}: {response_content}. Skipping."
            )
            continue

        all_tracks_df.loc[all_tracks_df["track_id"] == track_id, "mask"] = mask_value
    return all_tracks_df


def filter_results_by_mask(results, all_tracks_df):
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
