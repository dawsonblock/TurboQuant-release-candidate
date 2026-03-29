from mlx_lm import generate, load

model, tokenizer = load("mlx-community/SmolLM-135M-Instruct-4bit")

prompt = "Explain quantum computing in 3 sentences."
print("Running with dense cache first...")
dense_text = generate(model, tokenizer, prompt, max_tokens=100)
print(dense_text)

print("\nRunning with TurboQuant cache...")
tq_text = generate(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    turboquant_k_start=16,
    turboquant_main_bits=3,
    turboquant_group_size=64
)
print(tq_text)
