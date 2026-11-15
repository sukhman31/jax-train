import os, sys, time
import jax
import jax.numpy as jnp
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer
import pickle

CHECKPOINT_PATH = os.path.join(ROOT, 'checkpoints', 'gpt_shakespeare.pkl')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_model(checkpoint_path, rngs):
    log("Loading checkpoint...")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    config = checkpoint['config']
    tokenizer = BPETokenizer(max_merges=config['max_merges'])
    tokenizer.pairs_dict = checkpoint['tokenizer_pairs']
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    model = Model(config, rngs)
    nnx.update(model, checkpoint['model_state'])
    model.eval()
    log("Model loaded.")
    return model, tokenizer, config


def init_cache(config, batch_size, max_len):
    num_kv_heads = config.get('num_kv_heads', config['num_heads'])
    head_dim = config['output_dim'] // config['num_heads']
    cache = []
    for _ in range(config['num_of_blocks']):
        k = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))
        v = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))
        cache.append((k, v))
    return cache


def generate_no_cache(model, token_ids, rngs, max_new_tokens, max_seq_len):
    generated = list(token_ids)
    for _ in range(max_new_tokens):
        context = generated[-max_seq_len:]
        x = jnp.array([context])
        logits, _ = model(x, rngs)
        next_id = int(jnp.argmax(logits[0, -1, :]))
        generated.append(next_id)
    return generated


def generate_with_cache(model, token_ids, config, rngs, max_new_tokens):
    generated = list(token_ids)
    prompt_len = len(token_ids)
    max_len = prompt_len + max_new_tokens

    cache = init_cache(config, 1, max_len)

    x = jnp.array([token_ids])
    logits, cache = model(x, rngs, cache=cache, cache_pos=0)
    next_id = int(jnp.argmax(logits[0, -1, :]))
    generated.append(next_id)

    for i in range(max_new_tokens - 1):
        x = jnp.array([[next_id]])
        logits, cache = model(x, rngs, cache=cache, cache_pos=prompt_len + i)
        next_id = int(jnp.argmax(logits[0, -1, :]))
        generated.append(next_id)
    return generated


def build_jit_generate(model, rngs, config):
    graphdef, state = nnx.split(model)

    def decode_single(state, token_id, cache, cache_pos):
        m = nnx.merge(graphdef, state)
        x = token_id.reshape(1, 1)
        logits, new_cache = m(x, rngs, cache=cache, cache_pos=cache_pos)
        next_id = jnp.argmax(logits[0, 0, :]).astype(jnp.int32)
        return next_id, new_cache

    @jax.jit
    def prefill(state, token_ids_arr, cache):
        m = nnx.merge(graphdef, state)
        x = token_ids_arr.reshape(1, -1)
        logits, new_cache = m(x, rngs, cache=cache, cache_pos=0)
        first_id = jnp.argmax(logits[0, -1, :]).astype(jnp.int32)
        return first_id, new_cache

    def make_decode_loop(max_new_tokens, prompt_len):
        @jax.jit
        def decode_loop(state, first_id, cache):
            generated = jnp.zeros(max_new_tokens, dtype=jnp.int32)
            generated = generated.at[0].set(first_id)

            def body(i, carry):
                gen_tokens, c, prev_id = carry
                next_id, new_c = decode_single(state, prev_id, c, prompt_len + i)
                gen_tokens = gen_tokens.at[i].set(next_id)
                return gen_tokens, new_c, next_id

            generated, _, _ = jax.lax.fori_loop(
                1, max_new_tokens, body, (generated, cache, first_id)
            )
            return generated
        return decode_loop

    return state, prefill, make_decode_loop


def generate_jit_loop(state, prefill_fn, decode_loop_fn, token_ids, config, max_new_tokens):
    prompt_len = len(token_ids)
    max_len = prompt_len + max_new_tokens

    cache = init_cache(config, 1, max_len)
    token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
    first_id, cache = prefill_fn(state, token_ids_arr, cache)
    generated = decode_loop_fn(state, first_id, cache)
    return generated


def benchmark(label, fn, warmup=2, runs=5):
    log(f"  {label}: warming up ({warmup} runs)...")
    for i in range(warmup):
        t = time.perf_counter()
        fn()
        log(f"    warmup {i+1}/{warmup}: {time.perf_counter()-t:.3f}s")

    log(f"  {label}: timing ({runs} runs)...")
    times = []
    for i in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        log(f"    run {i+1}/{runs}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    return avg


def main():
    log(f"Device: {jax.devices()[0]}")
    print()

    rngs = nnx.Rngs(42)
    model, tokenizer, config = load_model(CHECKPOINT_PATH, rngs)

    prompt = "To be or not to be"
    token_ids = tokenizer.encode(prompt)
    max_seq_len = config.get('max_seq_len', 256)
    log(f"Prompt: \"{prompt}\" -> {len(token_ids)} tokens")
    print()

    state, prefill_fn, make_decode_loop = build_jit_generate(model, rngs, config)

    for num_tokens in [20, 50, 100]:
        log(f"=== Generating {num_tokens} tokens ===")

        no_cache_time = benchmark(
            "no_cache",
            lambda n=num_tokens: generate_no_cache(model, token_ids, rngs, n, max_seq_len),
        )
        no_cache_tps = num_tokens / no_cache_time

        cache_time = benchmark(
            "with_cache",
            lambda n=num_tokens: generate_with_cache(model, token_ids, config, rngs, n),
        )
        cache_tps = num_tokens / cache_time

        decode_loop_fn = make_decode_loop(num_tokens, len(token_ids))
        jit_time = benchmark(
            "jit_loop",
            lambda n=num_tokens: generate_jit_loop(
                state, prefill_fn, decode_loop_fn, token_ids, config, n
            ),
        )
        jit_tps = num_tokens / jit_time

        print()
        log(f"  RESULTS ({num_tokens} tokens):")
        log(f"    No cache:     {no_cache_time:.3f}s  ({no_cache_tps:.1f} tok/s)")
        log(f"    Cache (loop): {cache_time:.3f}s  ({cache_tps:.1f} tok/s)")
        log(f"    JIT + fori:   {jit_time:.3f}s  ({jit_tps:.1f} tok/s)")
        log(f"    Speedup vs no_cache: {no_cache_time/jit_time:.1f}x")
        print()


if __name__ == '__main__':
    main()
