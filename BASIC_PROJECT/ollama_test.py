# set OLLAMA_HOST env variable to 0.0.0.0:11434

from ollama import Client
client = Client(
  host='http://10.0.8.110:11434', #100.94.224.82:11434',
  headers={'Content-Type': 'application/json'}
)
response = client.chat(model='deepseek-r1:1.5b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response)