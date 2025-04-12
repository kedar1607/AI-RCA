# This code will consolidate all the logs from DataDog, CAL and FPTI 
# and save it in another table in chronological order. 
# E.g. this consolidates the logs for 2025-01-15 from 9 to 10 am.

import pandas as pd

df = None
# Load CSV files with proper datetime parsing
def consolidate_logs_in_time_window(start_time, end_time):
    global df
    df_dd = pd.read_csv("datadog_mobile_logs.csv", parse_dates=["timestamp"])
    df_be = pd.read_csv("backend_logs.csv", parse_dates=["timestamp"])
    df_as = pd.read_csv("analytics_logs.csv", parse_dates=["timestamp"])

    # Filter logs by the time window
    df_dd = df_dd[(df_dd["timestamp"] >= start_time) & (df_dd["timestamp"] <= end_time)]
    df_be = df_be[(df_be["timestamp"] >= start_time) & (df_be["timestamp"] <= end_time)]
    df_as = df_as[(df_as["timestamp"] >= start_time) & (df_as["timestamp"] <= end_time)]

    # Consolidate logs by merging on session_id, event_idx, and event_name.
    # This join will include BE logs only for events that have a backend call (i.e. where a correlation_id exists in DD).
    df_consolidated = pd.merge(
        df_dd,
        df_be,
        on=["session_id", "event_idx", "event_name"],
        how="left",
        suffixes=("_dd", "_be")
    )

    # Now merge the Analytics logs (which don't include correlation_id) on session_id, event_idx, and event_name.
    df_consolidated = pd.merge(
        df_consolidated,
        df_as,
        on=["session_id", "event_idx", "event_name"],
        how="left"
    )

    # Optional: sort the consolidated DataFrame by session_id and event_idx
    df_consolidated.sort_values(by=["session_id", "event_idx"], inplace=True)

    # Display a sample of the consolidated snapshot
    # print("Consolidated Logs Snapshot:")
    # print(df_consolidated.head(10))
    # Save the consolidated view to CSV if desired
    df_consolidated.to_csv("consolidated_logs.csv", index=False)
    df = df_consolidated
    return df_consolidated