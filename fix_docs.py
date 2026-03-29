import glob
import os

files_to_check = [
    "README.md",
    "docs/integration.md",
    "docs/evaluation.md",
    "docs/cache-format.md",
    "docs/architecture.md",
    "mlx_lm/generate.py",
    "turboquant/eval/perplexity.py",
    "integrations/mlx/upgrade.py",
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            text = f.read()
        
        # Replace module name
        new_text = text.replace("mlx_lm.cache_upgrade", "integrations.mlx.upgrade")
        
        # Replace TurboQuantKCache for production path (not the legacy adapter references if any, but replace in standard mentions where we mean KVCompressor)
        if file_path == "docs/integration.md":
            new_text = new_text.replace("KVCache with TurboQuantKCache", "KVCache with KVCompressor")
            new_text = new_text.replace("— TurboQuantKCache is fully", "— KVCompressor is fully")
            
        if text != new_text:
            with open(file_path, "w") as f:
                f.write(new_text)
            print(f"Updated {file_path}")
