from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor
from turboquant.runtime.attention import attention_kernel, maybe_turboquant_attention

class TurboQuantRuntime:
    def __init__(self, config: TurboQuantConfig):
        self.kv = KVCompressor(config)

    def step(self, keys, values):
        return self.kv.update_and_fetch(keys, values)

    def attention(self, queries, state):
        # We assume state gives back the rotated queries and K/V blocks
        return maybe_turboquant_attention(queries, state)
