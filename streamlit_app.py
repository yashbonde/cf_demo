import os

# USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("true", "1")

import streamlit as st
from time import time

from chainfury.components.qdrant import qdrant_read
from chainfury.components.openai import openai_chat, OpenAIChat
from chainfury.components.tune import chatnbx, ChatNBX

from load_data import get_embedding

COLLECTION_NAME = "blitzscaling"

def blitzscaling_chat_fn(
  question: str,
  model: str = ""
):

  # load the data points from the memory
  _st = time()
  st.write("Loading data points")
  embedding, err = get_embedding({"text": question})
  if err:
    return None, err
  out, err = qdrant_read(
    embeddings = embedding.tolist(),
    collection_name = COLLECTION_NAME,
    top = 3,
  )
  if err:
    return None, err

  # create a string with all the data points
  data_points = [x["payload"] for x in out["data"]]
  data_points_text = [x["text"] for x in data_points]
  dp_text = ""
  for i, text in enumerate(data_points_text):
      dp_text += f"<id>[{i}]</id>\n\n{text}"
      dp_text += "\n------\n"
  st.write(f"Loaded data points in {time() - _st:.2f} seconds")

  _st = time()
  messages=[{
        "role" : "system", 
        "content" : '''
You are a helpful assistant that is helping user summarize the information with citations.

Tag all the citations with tags around it like:

```
this is some text [<id>2</id>, <id>14</id>]
```'''},
      {
        "role": "user",
        "content": f'''
Data points collection:

{dp_text}

---

User has asked the following question:

{question}
'''}]

  if USE_OPENAI:
    st.write("Calling LLM")
    messages = [OpenAIChat.Message(**x) for x in messages]
    out = openai_chat(model = model, messages = messages)
  else:
    st.write("Calling [ChatNBX](https://chat.nbox.ai)")
    messages = [ChatNBX.Message(**x) for x in messages]
    out = chatnbx(model = model, messages = messages)

  try:
    response = out["choices"][0]["message"]["content"]
    st.write(f"Called LLM in {time() - _st:.2f} seconds")
  except:
    st.error(out)
    st.write(f"Called LLM in {time() - _st:.2f} seconds")
    return None, out

  return (response, data_points), None


# ------ script ------ #

st.title("Blitzscaling Q/A")
USE_OPENAI = st.toggle("Use OpenAI's `gpt-3.5-turbo`", value=False)

if USE_OPENAI:
  model = "gpt-3.5-turbo"
else:
  model = "llama-2-chat-70b-4k"
st.write(f'''This demo shows how to use [ChainFury](https://nimbleboxai.github.io/ChainFury/index.html)
to build a simple chatbot that can answer questions about blitzscaling. [Code](https://github.com/yashbonde/cf_demo).
This demo uses
{"[OpenAI](https://openai.com)" if USE_OPENAI else "[ChatNBX](https://chat.nbox.ai)"}'s `{model}` as the model.

- 📚 [Access the Blitzscaling PDF](https://drive.google.com/file/d/1QeWwfxEcYyAXkLexCgUX4AWr6nnO3Aqk/view?usp=sharing)
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
    result, err = blitzscaling_chat_fn(prompt, model)
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
