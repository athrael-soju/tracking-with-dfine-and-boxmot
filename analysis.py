import pandas as pd
from pathlib import Path
import os
import glob
import logging

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASET_DIR = Path("static/dataset")

def list_dataset_files():
    """Lists all CSV files in the dataset directory."""
    if not DATASET_DIR.exists():
        logger.warning(f"Dataset directory {DATASET_DIR} does not exist.")
        return []
    try:
        files = [f.name for f in DATASET_DIR.glob("*.csv") if f.is_file()]
        logger.info(f"Found dataset files: {files}")
        return files
    except Exception as e:
        logger.error(f"Error listing dataset files: {e}")
        return []

def load_dataset(dataset_filename: str):
    """Loads a dataset CSV file into a pandas DataFrame."""
    if not dataset_filename:
        logger.warning("No dataset filename provided for loading.")
        return None
    file_path = DATASET_DIR / dataset_filename
    if not file_path.exists():
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset: {dataset_filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_filename}: {e}")
        raise

def get_basic_insights(df: pd.DataFrame):
    """Generates basic numerical insights from the loaded dataset."""
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None. Cannot generate insights.")
        return {"error": "DataFrame is empty or None."}
    
    insights = {}
    try:
        insights["total_detections"] = int(len(df))
        insights["unique_classes_count"] = int(df["class_name"].nunique())
        insights["class_counts"] = df["class_name"].value_counts().astype(int).to_dict()
        
        if "timestamp_seconds" in df.columns:
            insights["max_timestamp_seconds"] = float(df["timestamp_seconds"].max())
            insights["min_timestamp_seconds"] = float(df["timestamp_seconds"].min())

        # Tracker-specific insights
        if "tracker_id" in df.columns:
            tracked_df = df[df["tracker_id"] != -1]
            if not tracked_df.empty and tracked_df["tracker_id"].nunique() > 0:
                insights["unique_tracks_count"] = int(tracked_df["tracker_id"].nunique())
                track_lengths = tracked_df.groupby("tracker_id").size()
                insights["average_track_duration_frames"] = float(track_lengths.mean())
                insights["min_track_duration_frames"] = int(track_lengths.min())
                insights["max_track_duration_frames"] = int(track_lengths.max())
            else:
                insights["unique_tracks_count"] = 0
                insights["average_track_duration_frames"] = 0.0
        else:
            insights["unique_tracks_count"] = 0
            insights["average_track_duration_frames"] = 0.0

        insights["confidence_mean"] = float(df["confidence"].mean())
        insights["confidence_median"] = float(df["confidence"].median())
        insights["confidence_std"] = float(df["confidence"].std())
        
        logger.info(f"Generated insights: {insights}")
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {"error": f"Error generating insights: {str(e)}"} 

def plot_class_distribution(df: pd.DataFrame):
    """Generates an interactive bar chart of class distribution using Plotly."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot generate class distribution plot.")
        return None
    if df is None or df.empty or "class_name" not in df.columns:
        return None
    try:
        class_counts = df["class_name"].value_counts().reset_index()
        class_counts.columns = ["class_name", "count"]
        fig = px.bar(class_counts, x="count", y="class_name", orientation='h', 
                     title="Class Distribution", labels={"count": "Count", "class_name": "Class Name"})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    except Exception as e:
        logger.error(f"Error generating class distribution plot: {e}")
        return None

def plot_track_length_histogram(df: pd.DataFrame):
    """Generates an interactive histogram of track lengths using Plotly."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot generate track length histogram.")
        return None
    if df is None or df.empty or "tracker_id" not in df.columns:
        logger.info("No tracker_id column for track length histogram.")
        return None
    
    tracked_df = df[df["tracker_id"] != -1]
    if tracked_df.empty or tracked_df["tracker_id"].nunique() == 0:
        logger.info("Not enough tracked data to generate track length histogram.")
        return None
    try:
        track_lengths = tracked_df.groupby("tracker_id").size().reset_index(name='length')
        fig = px.histogram(track_lengths, x="length", nbins=20, title="Track Length Distribution (Frames per Track)",
                           labels={"length": "Number of Frames in Track"})
        fig.update_layout(yaxis_title="Number of Tracks")
        return fig
    except Exception as e:
        logger.error(f"Error generating track length histogram: {e}")
        return None

def plot_object_trajectories(df: pd.DataFrame, max_tracks_to_plot=20):
    """Plots the trajectories of tracked objects using Plotly, with start/end markers."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot generate trajectory plot.")
        return None
    if df is None or df.empty or not all(col in df.columns for col in ["frame_index", "tracker_id", "x_min", "y_min", "x_max", "y_max"]):
        logger.info("Insufficient data for trajectory plot.")
        return None
    
    tracked_df = df[df["tracker_id"] != -1].copy()
    if tracked_df.empty:
        logger.info("No tracked objects to plot trajectories.")
        return None

    tracked_df["x_center"] = (tracked_df["x_min"] + tracked_df["x_max"]) / 2
    tracked_df["y_center"] = (tracked_df["y_min"] + tracked_df["y_max"]) / 2
    tracked_df["tracker_id"] = tracked_df["tracker_id"].astype(str)

    unique_track_ids = tracked_df["tracker_id"].unique()
    ids_to_plot = unique_track_ids
    if len(unique_track_ids) > max_tracks_to_plot:
        logger.info(f"Too many tracks ({len(unique_track_ids)}). Plotting trajectories for the first {max_tracks_to_plot} tracks by appearance.")
        first_occurrence_order = tracked_df.drop_duplicates(subset=["tracker_id"], keep='first')
        ids_to_plot = first_occurrence_order["tracker_id"].unique()[:max_tracks_to_plot]
        
    plot_df = tracked_df[tracked_df["tracker_id"].isin(ids_to_plot)]

    if plot_df.empty:
        logger.info("No tracks selected for plotting after filtering.")
        return None

    try:
        fig = px.line(plot_df, x="x_center", y="y_center", color="tracker_id", 
                      line_group="tracker_id", hover_name="tracker_id",
                      title=f"Object Trajectories (Max {max_tracks_to_plot} tracks)",
                      markers=False) # Turn off default markers for lines to add custom start/end

        for track_id_str in plot_df["tracker_id"].unique():
            track_data = plot_df[plot_df["tracker_id"] == track_id_str].sort_values(by="frame_index")
            if not track_data.empty:
                # Start marker
                fig.add_trace(go.Scatter(x=[track_data.iloc[0]["x_center"]], y=[track_data.iloc[0]["y_center"]],
                                          mode='markers', marker=dict(symbol='circle', size=10, color=px.colors.qualitative.Plotly[int(track_id_str) % len(px.colors.qualitative.Plotly) if track_id_str.isdigit() else 0]),
                                          name=f"Start {track_id_str}", legendgroup=track_id_str, showlegend=False))
                # End marker
                fig.add_trace(go.Scatter(x=[track_data.iloc[-1]["x_center"]], y=[track_data.iloc[-1]["y_center"]],
                                          mode='markers', marker=dict(symbol='x', size=10, color=px.colors.qualitative.Plotly[int(track_id_str) % len(px.colors.qualitative.Plotly) if track_id_str.isdigit() else 0]),
                                          name=f"End {track_id_str}", legendgroup=track_id_str, showlegend=False))

        fig.update_layout(
            xaxis_title="X Center Coordinate",
            yaxis_title="Y Center Coordinate",
            yaxis_autorange='reversed' 
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating trajectory plot: {e}")
        return None

def plot_unique_tracks_over_time(df: pd.DataFrame):
    """Plots the number of unique active tracks per frame using Plotly."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot generate unique tracks over time plot.")
        return None
    if df is None or df.empty or not all(col in df.columns for col in ["frame_index", "tracker_id"]):
        logger.info("Insufficient data for unique tracks over time plot.")
        return None

    tracked_df = df[df["tracker_id"] != -1]
    if tracked_df.empty:
        logger.info("No tracked objects to plot unique tracks over time.")
        return None

    try:
        unique_tracks_per_frame = tracked_df.groupby("frame_index")["tracker_id"].nunique().reset_index()
        unique_tracks_per_frame.columns=["frame_index", "unique_track_count"]
        
        fig = px.line(unique_tracks_per_frame, x="frame_index", y="unique_track_count", 
                      title="Number of Unique Active Tracks Over Time", 
                      labels={"frame_index": "Frame Index", "unique_track_count": "Number of Unique Tracks"},
                      markers=True)
        return fig
    except Exception as e:
        logger.error(f"Error generating unique tracks over time plot: {e}")
        return None

def get_track_summary_data(df: pd.DataFrame):
    """Generates a summary for each track (ID, class, start/end frame, duration)."""
    if df is None or df.empty or "tracker_id" not in df.columns:
        logger.info("No tracker_id column for track summary table.")
        return pd.DataFrame() # Return empty DataFrame

    tracked_df = df[df["tracker_id"] != -1].copy()
    if tracked_df.empty:
        logger.info("No tracked objects to summarize.")
        return pd.DataFrame()

    summary_list = []
    for track_id, group in tracked_df.groupby("tracker_id"):
        start_frame = group["frame_index"].min()
        end_frame = group["frame_index"].max()
        duration_frames = end_frame - start_frame + 1
        # Get class name - most frequent if multiple, or first if all same
        class_name = group["class_name"].mode().iloc[0] if not group["class_name"].mode().empty else "N/A"
        
        start_timestamp = group[group["frame_index"] == start_frame]["timestamp_seconds"].iloc[0] if "timestamp_seconds" in group.columns else "N/A"
        end_timestamp = group[group["frame_index"] == end_frame]["timestamp_seconds"].iloc[0] if "timestamp_seconds" in group.columns else "N/A"
        
        summary_list.append({
            "Track ID": track_id,
            "Class Name": class_name,
            "Start Frame": start_frame,
            "End Frame": end_frame,
            "Duration (Frames)": duration_frames,
            "Start Time (s)": start_timestamp,
            "End Time (s)": end_timestamp
        })
    
    summary_df = pd.DataFrame(summary_list)
    return summary_df.sort_values(by="Start Frame") 