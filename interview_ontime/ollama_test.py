from ollama import chat

messages = [
  {
    'role': 'user',
    'content': '根据内容判断当前语句是否是面试问题？忽略打招呼，寒暄，聊天，只判断问题是否为专业只是问题，只输出是或否。当前语句：是的是的，你好你好',
  },
]

response = chat('qwen3.5:4b', messages=messages, think=False)

# print('Thinking:\n========\n\n' + response.message.thinking)
print('\nResponse:\n========\n\n' + response.message.content)