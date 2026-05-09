import requests
import json
import sseclient
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Start chat...")
res = requests.post("http://localhost:8000/chat", json={"question": "luật giao thông"}, stream=True)
client = sseclient.SSEClient(res)
thread_id = None
for event in client.events():
    if event.data:
        data = json.loads(event.data)
        print("Chat event:", data)
        if data.get("status") == "pending":
            thread_id = data["thread_id"]

print("Thread ID:", thread_id)

if thread_id:
    print("Resume...")
    res2 = requests.post("http://localhost:8000/resume", json={"thread_id": thread_id, "action": "approve", "edited_content": "Test answer"}, stream=True)
    client2 = sseclient.SSEClient(res2)
    for event in client2.events():
        try:
            print("Resume event:", event.data)
        except UnicodeEncodeError:
            print("Resume event: [Unicode Error, but received data]")
