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
    """Display the main tracking interface with video processing and AI query functionality.

    Creates a comprehensive Streamlit interface for drone video tracking that includes:
    - AI-powered query interface for filtering specific objects
    - Real-time video processing with object detection and tracking
    - Side-by-side display of raw and annotated video feeds
    - CSV logging of tracking data for analysis

    The function processes video frames using YOLO object detection and tracking,
    applies LLM-based filtering based on user queries, and maintains persistent
    tracking data across the session.

    Args:
        model: Pre-loaded YOLO model instance for object detection and tracking.
        video_path (str): File path to the video file to be processed.

    Returns:
        None: This function renders Streamlit UI components directly.

    UI Components:
        - AI Query Interface: Text input and control buttons for filtering
        - Video Display: Raw video feed and annotated results side-by-side
        - Control Buttons:
          * Apply Query: Activates LLM filtering for specified objects
          * Stop: Halts video processing and clears query
          * New Log File: Creates fresh CSV file for new experiment
          * Start Detection: Begins video processing loop

    Workflow:
        1. User enters natural language query (e.g., "blue cars")
        2. Clicks Start Detection to begin video processing
        3. Each frame is processed with YOLO tracking
        4. LLM analyzes cropped objects against user query
        5. Results are filtered and displayed in annotated video
        6. Tracking data is continuously saved to CSV file

    Note:
        - Requires st.session_state.csv_filename to be initialized
        - Video processing runs in blocking loop until completion
        - Uses OpenCV for video handling and frame processing
        - Supports BotSORT tracker for object persistence

    Dependencies:
        - YOLO model for object detection/tracking
        - LLM model (Qwen2.5-VL) for object categorization
        - OpenCV for video processing
        - Streamlit for UI components
    """
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
