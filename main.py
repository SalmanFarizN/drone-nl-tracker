from src.utils import *
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import cv2
import time
import threading


def main():

    video_path = "data/sample_drone_low_alt.mp4"
    model = YOLO("models/yolov8n-VisDrone.pt")
    cap = cv2.VideoCapture(video_path)

    all_tracks_df = pd.DataFrame(
        {
            "track_id": pd.Series([], dtype="Int64"),  # Nullable integer
            "class_id": pd.Series([], dtype="int"),
            "class_name": pd.Series([], dtype="string"),
            "confidence": pd.Series([], dtype="float"),
            "bbox": pd.Series([], dtype="object"),  # For lists
            "mask": pd.Series([], dtype="Int64"),  # Nullable integer for NA support
        }
    )

    st.set_page_config(layout="wide")
    st.title("🚁 Drone Visual Tracking Dashboard")

    tabs = st.tabs(["🎥 Tracking", "📑 Logs"])

    with tabs[0]:
        # Chat interface box at the top
        with st.container():
            st.subheader("💬 AI Tracking Query")

            user_input = st.text_input(
                "What would you like to track?",
                placeholder="e.g., I want to track all blue cars",
                key="chat_input",
                help="Enter a description to filter objects using AI. Leave empty to show all objects.",
            )

            # Create columns for buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                apply_clicked = st.button("🔍 Apply Query", key="apply_btn")
            with col2:
                clear_clicked = st.button("❌ Clear", key="clear_btn")

            # Determine the active query
            user_query = ""
            if apply_clicked and user_input.strip():
                user_query = user_input.strip()
            elif not clear_clicked and user_input.strip():
                # Auto-apply when typing (like hitting Enter)
                user_query = user_input.strip()

            if user_query:
                st.success(f"🎯 Active query: **{user_query}**")
            else:
                st.info("🔍 No query - will show all detected objects")

        st.divider()  # Visual separator

        # Video display columns
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

                if all_tracks_df.empty:
                    all_tracks_df = current_df
                else:
                    all_tracks_df = update_all_tracks(all_tracks_df, current_df)

                # Only do LLM categorization and filtering if user has entered a query
                if user_query and user_query.strip():
                    all_tracks_df = categorize_tracks_with_llm(
                        all_tracks_df, results, user_query
                    )
                    # Filtering logic
                    filtered_results = filter_results_by_mask(results, all_tracks_df)
                    annotated_frame = filtered_results.plot()
                else:
                    # No user query - show all detections
                    annotated_frame = results[0].plot()

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_window.image(annotated_frame, channels="RGB")

            cap.release()
            cv2.destroyAllWindows()

    with tabs[1]:
        st.subheader("Tracking Logs")
        st.dataframe(all_tracks_df, width="stretch")


if __name__ == "__main__":
    main()
