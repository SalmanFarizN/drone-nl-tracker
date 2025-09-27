from src.utils import *
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import cv2
import time
import threading


def main():

    video_path = "data/sample_drone_vis.mp4"
    model = YOLO("models/yolov8n-VisDrone.pt")
    cap = cv2.VideoCapture(video_path)

    all_tracks_df = pd.DataFrame(
        {
            "track_id": pd.Series([], dtype="Int64"),  # Nullable integer
            "class_id": pd.Series([], dtype="int"),
            "class_name": pd.Series([], dtype="string"),
            "confidence": pd.Series([], dtype="float"),
            "bbox": pd.Series([], dtype="object"),  # For lists
            "mask": pd.Series([], dtype="int"),  # 0 or 1
        }
    )

    st.set_page_config(layout="wide")
    st.title("🚁 Drone Visual Tracking Dashboard")

    tabs = st.tabs(["🎥 Tracking", "📑 Logs", "💬 Chat"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Video")
            raw_window = st.empty()
        with col2:
            st.subheader("Annotated Video")
            annotated_window = st.empty()

        if st.button("▶️ Start Video"):
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                real_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_window.image(real_frame, channels="RGB")

                results = model.track(frame, persist=True, tracker="botsort.yaml")

                # Update logs dataframe
                current_df = extract_track_info(results[0])

                if all_tracks_df is None:
                    all_tracks_df = current_df
                else:
                    all_tracks_df = update_all_tracks(all_tracks_df, current_df)

                all_tracks_df = categorize_tracks_with_llm(all_tracks_df, results)

                # Filtering logic
                filtered_results = filter_results_by_mask(results, all_tracks_df)
                annotated_frame = filtered_results.plot()

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                annotated_window.image(annotated_frame, channels="RGB")

            cap.release()
            cv2.destroyAllWindows()

    with tabs[1]:
        st.subheader("Tracking Logs")
        st.dataframe(all_tracks_df, width="stretch")


if __name__ == "__main__":
    main()
