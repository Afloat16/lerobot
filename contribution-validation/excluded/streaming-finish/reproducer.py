"""Reproduce finish_episode waiting on a full queue after a real PyAV error."""

import json
import queue
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import StreamingVideoEncoder


with tempfile.TemporaryDirectory() as temp:
    encoder = StreamingVideoEncoder(
        fps=30,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=30),
        queue_maxsize=1,
        encoder_threads=1,
    )
    encoder.start_episode(["camera"], Path(temp))
    worker = encoder._threads["camera"]
    frames = encoder._frame_queues["camera"]
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_open = av.open
    errors = []

    # An odd width is rejected by the actual libx264 yuv420p encoder.

    def gated_open(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("probe synchronization failed")
        return original_open(*args, **kwargs)

    def finish():
        try:
            encoder.finish_episode()
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        finally:
            finished.set()

    with patch("lerobot.datasets.video_utils.av.open", gated_open):
        encoder.feed_frame("camera", np.zeros((64, 95, 3), dtype=np.uint8))
        assert entered.wait(5), "worker never reached PyAV"
        encoder.feed_frame("camera", np.zeros((64, 95, 3), dtype=np.uint8))
        assert frames.full()
        release.set()
        worker.join(5)
        assert not worker.is_alive(), "worker did not report the actual I/O error"
        finalizer = threading.Thread(target=finish, daemon=True)
        finalizer.start()
        returned_before_cleanup = finished.wait(1)
        # Always release the baseline's blocked producer before exiting the probe.
        if not returned_before_cleanup:
            frames.get_nowait()
        finalizer.join(5)
        assert not finalizer.is_alive()
        encoder.close()

    print(json.dumps({
        "returned_without_manual_queue_drain": returned_before_cleanup,
        "errors": errors,
        "actual_encoder_worker_exited": not worker.is_alive(),
        "pyav_version": av.__version__,
    }, indent=2))
