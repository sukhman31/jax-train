"""Stream a generation from a running inference server: python -m serving.client "Once upon a time" """
import sys

import grpc

from serving.gen.inference import inference_api_pb2 as pb
from serving.gen.inference import inference_api_pb2_grpc as pb_grpc


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'Once upon a time'
    target = sys.argv[2] if len(sys.argv) > 2 else 'localhost:50051'
    stub = pb_grpc.InferenceServiceStub(grpc.insecure_channel(target))
    print(prompt, end='', flush=True)
    for msg in stub.Generate(pb.GenerateRequest(prompt=prompt, max_new_tokens=100)):
        if msg.done:
            s = msg.stats
            print(f'\n\n[{s.prompt_tokens} prompt tok, {s.generated_tokens} generated, '
                  f'ttft {s.ttft_ms:.0f} ms, {s.tokens_per_second:.1f} tok/s]')
        else:
            print(msg.text, end='', flush=True)


if __name__ == '__main__':
    main()
