import os

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("true", "1")

import streamlit as st
from time import time

from chainfury import memory_registry
from chainfury.components.openai import openai_chat, OpenAIChat

COLLECTION_NAME = "blitzscaling"

def blitzscaling_chat_fn(
  question: str,
  model: str = "gpt-3.5-turbo"
):
  # load the data points from the memory
  _st = time()
  st.write("Loading data points")
  mem  = memory_registry.get_read("qdrant")
  out, err = mem({
    "items": [question],
    "embedding_model": "openai-embedding",
    "limit": 1,
    "collection_name": COLLECTION_NAME
  })
  if err:
    return None, err

  # create a string with all the data points
  data_points = [x["payload"] for x in out["items"]["data"]]
  data_points_text = [x["text"] for x in data_points]
  dp_text = ""
  for i, text in enumerate(data_points_text):
      dp_text += f"<id>[{i}]</id>\n\n{text}"
      dp_text += "\n------\n"
  st.write(f"Loaded data points in {time() - _st:.2f} seconds")

  _st = time()
  st.write("Calling LLM")
  out = openai_chat(
    model = model,
    messages=[
      OpenAIChat.Message(
        role = "system", 
        content = '''
You are a helpful assistant that is helping user summarize the information with citations.

Tag all the citations with tags around it like:

```
this is some text [<id>2</id>, <id>14</id>]
```'''),
      OpenAIChat.Message(
        role = "user",
        content = f'''
Data points collection:

{dp_text}

---

User has asked the following question:

{question}
''')]
  )

  response = out["choices"][0]["message"]["content"]
  st.write(f"Called LLM in {time() - _st:.2f} seconds")

  return (response, data_points), None


# ------ script ------ #

st.title("Blitzscaling Q/A")
st.write('''This demo shows how to use [ChainFury](https://nimbleboxai.github.io/ChainFury/index.html)
         to build a simple chatbot that can answer questions about blitzscaling. [Code](https://github.com/yashbonde/cf_demo)
''')

@st.cache_resource
def Chat():
  return {}

@st.cache_resource
def ChatMode():
  return [False]

# chat = Chat()
# chat_modes = ChatMode()

prompt = st.chat_input("Ask it question on Blitzscaling")
if prompt:
  usr_msg = st.chat_message("user")
  usr_msg.write(prompt)

  with st.status("🦋 effect", expanded = True) as status:
    # if chat_modes[-1]:
    #   messages = [OpenAIChat.Message(role = "system", content = 'You are a helpful assistant trying to answer users question')]
    #   for i, (prompt, response, data_points) in enumerate(chat):
    #     messages.append(OpenAIChat.Message(role = "user", content = prompt))
    #     messages.append(OpenAIChat.Message(role = "assistant", content = response))
    #   out = openai_chat(model = "gpt-3.5-turbo", messages = messages)
    #   response = out["choices"][0]["message"]["content"]
    #   result = (response, "This was in chat mode.")
    # else:

    result, err = blitzscaling_chat_fn(prompt)
    if err:
      status.update(label="Error!", state="error", expanded=True)
      st.error(err)
    else:
      response, data_points = result
      # chat.append((prompt, response, data_points))
      status.update(label="Chain complete!", state="complete", expanded=False)

  ast_msg = st.chat_message("assistant")
  ast_msg.write(response)
  with st.expander("Citations"):
    st.write(data_points)

  # # iterate over the chat
  # for prompt, response, data_points in chat:
  #   # write the users message
  #   msg = st.chat_message("user")
  #   msg.write(prompt)
    
  #   # write the systems message
  #   msg = st.chat_message("assistant")
  #   msg.write(response)

  #   # write the citations for the chat
  #   with st.expander("Citations"):
  #     st.write(data_points)
