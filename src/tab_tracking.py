from src.file_io import create_csv_file
from src.data_process import (
    update_all_tracks,
    extract_track_info,
    filter_results_by_mask,
)
from src.vlm_analysis import categorize_tracks_with_llm
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import cv2


def tab_tracking(model, video_path) -> None:
    # Chat interface box at the top
    with st.container():
        st.subheader("💬 AI Tracking Query")

        user_input = st.text_input(
            "What would you like to track?",
            placeholder="e.g., Track all blue cars",
            key="chat_input",
            help="Enter a description to filter objects using AI. Leave empty to show all objects.",
        )

        # Create columns for buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            apply_clicked = st.button("🔍 Apply Query", key="apply_btn")
        with col2:
            clear_clicked = st.button("❌ Stop", key="clear_btn")
        with col3:
            csv_clicked = st.button("📁 New Log File", key="csv_btn")

        # Determine the active query
        user_query = ""
        if apply_clicked and user_input.strip():
            user_query = user_input.strip()
        elif not clear_clicked and user_input.strip():
            user_query = user_input.strip()

        if clear_clicked:
            st.rerun()

        if csv_clicked:
            st.session_state.csv_filename = create_csv_file()
            st.rerun()

        if user_query:
            st.success(f"🎯 Active query: **{user_query}**")
        else:
            st.info("🔍 No query - will show all detected objects")

    st.divider()

    # Video display columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Video")
        raw_window = st.empty()
    with col2:
        st.subheader("Annotated Video")
        annotated_window = st.empty()

    if st.button("▶️ Start Detection"):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            real_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_window.image(real_frame, channels="RGB")

            results = model.track(frame, persist=True, tracker="botsort.yaml")

            # Update session state DataFrame (this persists across reruns)
            current_df = extract_track_info(results[0])

            all_tracks_df = pd.read_csv(st.session_state.csv_filename)

            if all_tracks_df.empty:
                all_tracks_df = current_df
            else:
                all_tracks_df = update_all_tracks(all_tracks_df, current_df)

            # Apply LLM filtering if query exists
            if user_query and user_query.strip():
                all_tracks_df = categorize_tracks_with_llm(
                    all_tracks_df, results, user_query
                )
                filtered_results = filter_results_by_mask(results, all_tracks_df)
                annotated_frame = filtered_results.plot()
            else:
                annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            annotated_window.image(annotated_frame, channels="RGB")

            # Write updated DataFrame to the unique CSV file
            all_tracks_df.to_csv(st.session_state.csv_filename, index=False)

        cap.release()
        cv2.destroyAllWindows()
