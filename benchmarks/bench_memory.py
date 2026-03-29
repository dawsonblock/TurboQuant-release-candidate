import mlx.core as mx
from turboquant import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor

def run_bench():
    config = TurboQuantConfig(head_dim=128, n_heads=32)
    
    print("Memory Benchmark (estimated):")
    # Native
    print("Native KV Cache: float16, 1 token -> 128 * 32 * 2 bytes")
    # TQ
    print("TurboQuant Cache: int8/int4 compressed bytes footprint")
    
if __name__ == "__main__":
    run_bench()
