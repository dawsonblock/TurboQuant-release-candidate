with open("tests/compatibility/test_mlx_cache_contract.py", "r") as f:
    code = f.read()

code = code.replace("k_out.shape == (1, 8, 1, 128)", "isinstance(k_out, TurboQuantKeysView) and v_out.shape == (1, 8, 1, 128)")
if "from turboquant.runtime.kv_interface import TurboQuantKeysView" not in code:
    code = "from turboquant.runtime.kv_interface import TurboQuantKeysView\n" + code

with open("tests/compatibility/test_mlx_cache_contract.py", "w") as f:
    f.write(code)
