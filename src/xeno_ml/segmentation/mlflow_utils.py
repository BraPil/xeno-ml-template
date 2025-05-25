"""Light wrapper so segmentation code stays readable."""
import os
import mlflow

# default to local ./mlruns folder if user has not set anything
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

def new_run(name: str):
    """Context-manager → with new_run('segmentation'): …"""
    return mlflow.start_run(run_name=name)
