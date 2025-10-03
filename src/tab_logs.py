import streamlit as st
import pandas as pd
import time


def tab_logs() -> None:
    """Display the tracking logs tab with real-time data updates.

    Creates a Streamlit interface for viewing drone tracking experiment data
    with automatic refresh functionality. Displays the tracking DataFrame
    and computed statistics including vehicle counts and classification metrics.

    The function reads tracking data from the CSV file specified in session state
    and presents it in a user-friendly format with key metrics displayed as
    Streamlit metric widgets.

    Returns:
        None: This function renders Streamlit UI components directly.

    UI Components:
        - DataFrame display: Shows all tracking data in tabular format
        - Unique vehicles detected: Total count of detected objects
        - Unique Classes: Number of different object classes found
        - Filtered Vehicles: Count of objects matching LLM query filters

    Note:
        - Auto-refreshes every 2 seconds using st.rerun()
        - Requires st.session_state.csv_filename to be set
        - Handles missing columns gracefully with fallback values
        - Uses st.empty() placeholder for smooth UI updates

    Warning:
        The auto-refresh mechanism can cause performance issues with large
        datasets or slow file I/O operations.
    """
    st.subheader("Tracking Logs")

    # Auto-refresh every 2 seconds when on logs tab
    placeholder = st.empty()

    with placeholder.container():
        if st.session_state.csv_filename:
            df = pd.read_csv(st.session_state.csv_filename)
            st.dataframe(df, width="stretch")

            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique vehicles detected", len(df))
            with col2:
                unique_classes = (
                    df["class_name"].nunique() if "class_name" in df.columns else 0
                )
                st.metric("Unique Classes", unique_classes)
            with col3:
                filtered_count = df["mask"].sum()
                st.metric(
                    "Filtered Vehicles",
                    filtered_count if pd.notna(filtered_count) else 0,
                )

    time.sleep(2)
    st.rerun()
