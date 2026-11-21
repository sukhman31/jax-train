import logging
import os
import resource
import threading
import time
from concurrent import futures

import grpc

from serving.engine import InferenceEngine, TextStreamer
from serving.gen.inference import inference_api_pb2 as pb
from serving.gen.inference import inference_api_pb2_grpc as pb_grpc

log = logging.getLogger('inference')

MAX_PROMPT_CHARS = 1000
MAX_NEW_TOKENS = 200
READY_TIMEOUT_S = 120
QUEUE_TIMEOUT_S = 30


def _clamp(value, default, lo, hi):
    return min(max(value or default, lo), hi)


class InferenceServicer(pb_grpc.InferenceServiceServicer):
    def __init__(self, checkpoint_path):
        self.engine = None
        self.ready = threading.Event()
        self.busy = threading.Lock()
        threading.Thread(target=self._load, args=(checkpoint_path,), daemon=True).start()

    def _load(self, checkpoint_path):
        start = time.perf_counter()
        engine = InferenceEngine(checkpoint_path)
        loaded = time.perf_counter()
        engine.warmup()
        done = time.perf_counter()
        self.engine = engine
        self.ready.set()
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info('model ready: load %.1fs, warmup %.1fs, peak RSS %.0f MB',
                 loaded - start, done - loaded, peak_mb)

    def Generate(self, request, context):
        prompt = request.prompt
        if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f'prompt must be 1-{MAX_PROMPT_CHARS} characters')

        if not self.ready.wait(READY_TIMEOUT_S):
            context.abort(grpc.StatusCode.UNAVAILABLE, 'model is still loading, try again shortly')
        engine = self.engine

        params = dict(
            max_new_tokens=_clamp(request.max_new_tokens, 150, 1, MAX_NEW_TOKENS),
            temperature=_clamp(request.temperature, 0.8, 0.05, 2.0),
            top_k=_clamp(request.top_k, 50, 1, engine.config['vocab_size']),
            top_p=_clamp(request.top_p, 0.9, 0.05, 1.0),
            repetition_penalty=_clamp(request.repetition_penalty, 1.2, 1.0, 2.0),
        )

        if not self.busy.acquire(timeout=QUEUE_TIMEOUT_S):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'model is busy, try again in a moment')
        try:
            yield from self._generate(engine, prompt, params, context)
        finally:
            self.busy.release()

    def _generate(self, engine, prompt, params, context):
        start = time.perf_counter()
        prompt_ids = engine.encode(prompt)
        streamer = TextStreamer(engine.tokenizer.vocab)
        first_token_at = None
        generated = 0

        for token_id in engine.stream_ids(prompt_ids, **params):
            if not context.is_active():
                log.info('client went away after %d tokens', generated)
                return
            generated += 1
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text = streamer.push(token_id)
            if text:
                yield pb.GenerateResponse(text=text)
            if streamer.stopped:
                break

        tail = streamer.finish()
        if tail:
            yield pb.GenerateResponse(text=tail)

        end = time.perf_counter()
        ttft_ms = (first_token_at - start) * 1000
        decode_s = end - first_token_at
        tok_per_s = (generated - 1) / decode_s if generated > 1 and decode_s > 0 else 0.0
        log.info('prompt %d tok, generated %d tok, ttft %.0f ms, %.1f tok/s',
                 len(prompt_ids), generated, ttft_ms, tok_per_s)
        yield pb.GenerateResponse(done=True, stats=pb.GenerationStats(
            prompt_tokens=len(prompt_ids),
            generated_tokens=generated,
            ttft_ms=ttft_ms,
            tokens_per_second=tok_per_s,
        ))


def serve():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    port = os.getenv('GRPC_PORT', '50051')
    checkpoint_path = os.getenv('CHECKPOINT_PATH', 'checkpoints/model.pkl')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(checkpoint_path), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    log.info('inference server listening on %s, loading %s', port, checkpoint_path)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
