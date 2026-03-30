import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
import gradio as gr
import os

os.environ["TQ_USE_METAL"] = "1"

model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name)

def chat_stream(message, history, max_tokens, temperature, k_bits, group_size):
    messages = []
    # history is a list of tuples: (user_msg, bot_msg)
    for h in history:
        if isinstance(h, dict):
            if h.get("role") == "user":
                messages.append({"role": "user", "content": h.get("content", "")})
            else:
                messages.append({"role": "assistant", "content": h.get("content", "")})
        else:
            if h[0]:
                messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})
                
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    response_text = ""
    
    generator = stream_generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=int(max_tokens), 
        temp=float(temperature),
        turboquant_k_start=0,
        turboquant_main_bits=int(k_bits),
        turboquant_group_size=int(group_size)
    )
    
    for response in generator:
        if response.text:
            response_text += response.text
            yield response_text

demo = gr.ChatInterface(
    fn=chat_stream,
    title="TurboQuant Streaming Chat",
    description="Interact with the LLM via Metal-Accelerated KV Cache! Try adjusting the compression live without resetting cache.",
    additional_inputs=[
        gr.Slider(minimum=10, maximum=1024, value=512, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=2, maximum=8, value=3, step=1, label="KV Cache k_bits"),
        gr.Dropdown(choices=[32, 64, 128], value=64, label="Group Size (Hardware Alignment)"),
    ]
)

if __name__ == "__main__":
    demo.launch(server_port=7860)