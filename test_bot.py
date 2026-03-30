import chat_ui
from chat_ui import bot_action
history = [{"role": "user", "content": "Hello!"}]
for h, s in bot_action(history, 512, 0.7, 3, 64):
    print("tick")
