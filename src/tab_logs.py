import streamlit as st
import pandas as pd
import time


def tab_logs() -> None:
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
