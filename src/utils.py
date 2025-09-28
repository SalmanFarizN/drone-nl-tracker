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
    # Add mask column to current_df NaN
    df["mask"] = np.nan
    return df


def update_all_tracks(all_tracks_df, current_df):
    """
    Update all_tracks_df with current frame data:
    - Add new track_ids
    - Update existing track_ids with new information
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


def encode_crop_for_qwen(orig_img, bbox, min_side=28, margin=8):
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)
    # expand bbox a bit for context
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(orig_img.shape[1], x2 + margin)
    y2 = min(orig_img.shape[0], y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = orig_img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None

    # Ensure shortest side >= min_side
    if min(h, w) < min_side:
        scale = float(min_side) / max(1, min(h, w))
        new_w = max(int(round(w * scale)), min_side)
        new_h = max(int(round(h * scale)), min_side)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ok, enc = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return enc.tobytes()


# Function to loop through all uncategorized tracks and ask the LLM to categorize them
def categorize_tracks_with_llm(all_tracks_df, result, query):
    non_categorized_tracks = all_tracks_df[all_tracks_df["mask"].isna()]

    if non_categorized_tracks.empty:
        print("No uncategorized tracks found.")
        return all_tracks_df

    for _, track in non_categorized_tracks.iterrows():
        track_id = track["track_id"]
        bbox = np.array(track["bbox"]).astype(int)
        x1, y1, x2, y2 = bbox

        orig_img = result[0].orig_img

        # For gemma
        # img = orig_img[y1:y2, x1:x2]

        # Convert the NumPy array (BGR not the RGB) to bytes
        # _, img_encoded = cv2.imencode(".jpg", img)  # Encode as JPEG
        # img_bytes = img_encoded.tobytes()  # Convert to bytes

        # Safe encode for qwen2.5-vl (guarantee min side >= 28)
        img_bytes = encode_crop_for_qwen(orig_img, bbox, min_side=28, margin=8)
        if img_bytes is None:
            print(f"Track ID {track_id}: crop too small/invalid, skipped.")
            continue

        # Define the system prompt
        system_prompt = {
            "role": "system",
            "content": "You are an expert in inferring characteristics (color, condition, type etc.) of vehicles from cropped, low-resolution drone images of vehicles from an overhead birds-eye view. Does the image match the description provided by the user? Please provide an answer as an int either 0 (if FALSE) or 1 (TRUE) for the image. Only respond with 1, if the image clearly matches the description. For colour, only asnwer with 1 if the majority of the car is clearly of that colour. Answer with 1 only if you are certain, otherwise answer with 0.",
        }

        # User query
        user_query = {
            "role": "user",
            "content": query,
            "images": [img_bytes],
        }

        # Send the prompts to the model
        res = ollama.chat(model="qwen2.5vl:7b", messages=[system_prompt, user_query])

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
