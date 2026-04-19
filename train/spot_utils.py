"""AWS spot instance resilience for long-running training jobs.

Training ImageNet models on spot instances requires:
1. Detecting spot interruption notices (2-minute warning)
2. Saving checkpoints to durable storage (S3) on interruption
3. Resuming from the latest checkpoint on a new instance
4. Auto-shutdown when all jobs complete (to avoid idle charges)

The spot interruption notice comes from the EC2 metadata endpoint.
We poll it in a background thread and call a user-provided save
function when an interruption is detected.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Optional
from urllib.error import URLError
from urllib.request import urlopen


# How often to poll the EC2 metadata endpoint (seconds)
_POLL_INTERVAL = 5

# EC2 metadata endpoint for spot interruption notices
_SPOT_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

# IMDSv2 token endpoint
_TOKEN_URL = "http://169.254.169.254/latest/api/token"


def _get_imds_token(ttl: int = 60) -> Optional[str]:
    """Get an IMDSv2 token. Returns None if not on EC2."""
    try:
        import urllib.request
        req = urllib.request.Request(_TOKEN_URL, method="PUT")
        req.add_header("X-aws-ec2-metadata-token-ttl-seconds", str(ttl))
        with urlopen(req, timeout=2) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _check_spot_interruption() -> Optional[dict]:
    """Check if a spot interruption notice has been issued.

    Returns the action dict if an interruption is pending, None otherwise.
    On non-EC2 machines this always returns None.
    """
    try:
        import urllib.request
        req = urllib.request.Request(_SPOT_ACTION_URL)
        token = _get_imds_token()
        if token:
            req.add_header("X-aws-ec2-metadata-token", token)
        with urlopen(req, timeout=2) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def setup_spot_handler(save_fn: Callable[[], None],
                       poll_interval: float = _POLL_INTERVAL) -> threading.Thread:
    """Register a background thread that saves a checkpoint on spot interruption.

    The save_fn is called once when a spot interruption notice is detected.
    After saving, SIGTERM is sent to the current process so the training
    loop can exit cleanly.

    Args:
        save_fn: Callable that saves the current checkpoint. Should be
            idempotent and fast (you have ~2 minutes).
        poll_interval: Seconds between metadata endpoint polls.

    Returns:
        The daemon thread (already started). You don't need to join it.
    """
    def _poll_loop():
        while True:
            action = _check_spot_interruption()
            if action is not None:
                print(f"\n[spot_utils] Spot interruption detected: {action}", flush=True)
                print("[spot_utils] Saving checkpoint...", flush=True)
                try:
                    save_fn()
                    print("[spot_utils] Checkpoint saved.", flush=True)
                except Exception as e:
                    print(f"[spot_utils] ERROR saving checkpoint: {e}", flush=True)
                # Give the training loop a chance to exit cleanly
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(poll_interval)

    thread = threading.Thread(target=_poll_loop, daemon=True, name="spot-handler")
    thread.start()
    return thread


def sync_to_s3(local_dir: str, s3_uri: str, quiet: bool = True) -> int:
    """Sync a local directory to S3 using the AWS CLI.

    Args:
        local_dir: Local directory path.
        s3_uri: S3 URI (e.g. s3://bucket/prefix/).
        quiet: Suppress per-file output.

    Returns:
        Return code from aws s3 sync (0 = success).
    """
    cmd = ["aws", "s3", "sync", local_dir, s3_uri]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[spot_utils] S3 sync failed: {result.stderr.strip()}", flush=True)
    return result.returncode


def sync_from_s3(s3_uri: str, local_dir: str, quiet: bool = True) -> int:
    """Download the latest checkpoint from S3 to a local directory.

    Args:
        s3_uri: S3 URI (e.g. s3://bucket/prefix/).
        local_dir: Local directory to download into.
        quiet: Suppress per-file output.

    Returns:
        Return code from aws s3 sync (0 = success).
    """
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_uri, local_dir]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[spot_utils] S3 download failed: {result.stderr.strip()}", flush=True)
    return result.returncode


def auto_shutdown_when_done():
    """Shutdown the instance after all jobs complete.

    Calls `sudo shutdown -h now`. Only works on Linux with sudo
    configured for passwordless shutdown (typical on EC2).
    """
    print("[spot_utils] All jobs complete. Shutting down instance...", flush=True)
    subprocess.run(["sudo", "shutdown", "-h", "now"])
