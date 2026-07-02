# PyWorker for Vast Serverless: wraps the LTX model_server (127.0.0.1:8188, POST /generate) so the
# Vast autoscaler can route + scale it. Detects readiness from the model-server log, proxies /generate,
# and warms the resident model with a startup benchmark.
import base64, os
from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

_TEST_IMG = base64.b64encode(open(os.path.join(os.path.dirname(__file__), "test.png"), "rb").read()).decode()
_NEG = "worst quality, blurry, morphing, warping, distortion, deformed, jitter, flicker, text, watermark, unnatural motion"

def _benchmark_payload():
    return {
        "prompt": "the scene is almost perfectly still, only the very faintest natural movement, the camera is locked and completely still",
        "negative_prompt": _NEG, "image_b64": _TEST_IMG, "steps": 8, "num_frames": 97, "height": 512, "width": 768,
    }

Worker(WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8188,
    model_log_file=os.environ.get("MODEL_LOG", "/var/log/portal/ltx.log"),
    handlers=[HandlerConfig(
        route="/generate",
        allow_parallel_requests=False,             # one GPU render at a time per worker
        max_queue_time=1800.0,                     # a queued clip can wait a while on a busy worker
        workload_calculator=lambda payload: 100.0, # every clip ~ equal cost -> constant metric
        benchmark_config=BenchmarkConfig(generator=_benchmark_payload, runs=2, concurrency=1),
    )],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete", "PIPELINE_READY"],
        on_error=["Traceback", "CUDA out of memory", "RuntimeError"],
    ),
)).run()
