import ollama
from src.utils import encode_crop_for_qwen
import numpy as np


def categorize_tracks_with_llm(all_tracks_df, result, query):
    """Categorize uncategorized tracks using LLM vision model analysis.

    Processes all tracks with NaN mask values by sending cropped vehicle images
    to the Qwen vision model along with a user query. The LLM determines if each
    vehicle matches the query description and updates the mask accordingly.

    Args:
        all_tracks_df (pd.DataFrame): Master tracking DataFrame with mask column.
        result (list): YOLO detection results containing original image data.
        query (str): User description to match against detected vehicles
            (e.g., "blue cars", "damaged vehicles").

    Returns:
        pd.DataFrame: Updated DataFrame with mask values set to:
            - 1: Vehicle matches the query description
            - 0: Vehicle does not match the query description
            - NaN: Unchanged (if processing failed)

    Note:
        Uses Qwen2.5-VL model via Ollama for image analysis. Requires proper
        model setup and sufficient image quality for accurate classification.
    """
    non_categorized_tracks = all_tracks_df[all_tracks_df["mask"].isna()]

    if non_categorized_tracks.empty:
        print("No uncategorized tracks found.")
        return all_tracks_df

    for _, track in non_categorized_tracks.iterrows():
        track_id = track["track_id"]
        bbox = np.array(track["bbox"]).astype(int)
        x1, y1, x2, y2 = bbox

        orig_img = result[0].orig_img

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
