from src.file_io import create_csv_file
from src.tab_tracking import tab_tracking
from src.tab_logs import tab_logs

import streamlit as st
from ultralytics import YOLO
import pandas as pd


def main():
    video_path = "data/sample_drone_low_alt_long.mp4"
    model = YOLO("models/yolov8n-VisDrone.pt")

    # Create a CSV file ONCE at the start of the app
    if "csv_filename" not in st.session_state:
        st.session_state.csv_filename = create_csv_file()

    st.set_page_config(layout="wide")
    st.title("🚁 Drone Visual Tracking Dashboard")

    tabs = st.tabs(["🎥 Tracking", "📑 Logs"])

    with tabs[0]:
        tab_tracking(model, video_path)

    with tabs[1]:
        tab_logs()


if __name__ == "__main__":
    main()
