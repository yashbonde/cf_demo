import fitz # PyMuPDF
import numpy as np
from fire import Fire
from tqdm import trange

from chainfury.utils import threaded_map
from chainfury.components.openai import openai_embedding
from chainfury.components.qdrant import _get_qdrant_client

from qdrant_client import models

client = _get_qdrant_client()

def disable_indexing(collection_name):
  return client.update_collection(
    collection_name=collection_name,
    optimizer_config=models.OptimizersConfigDiff(
      indexing_threshold=0
    )
  )

def recreate_collection(collection_name, embedding_dim):
  return client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
    optimizers_config=models.OptimizersConfigDiff(
      indexing_threshold=0,
    ),
  )

def enable_indexing(collection_name):
  return client.update_collection(
    collection_name=collection_name,
    optimizer_config=models.OptimizersConfigDiff(
      indexing_threshold=20000
    )
  )


def get_embedding(item, pbar = None):
  all_strings = [item["text"]]

  try:
    out = openai_embedding("text-embedding-ada-002", all_strings)
  except Exception as e:
    return None, item

  # create a payload
  _arr = np.array([x["embedding"] for x in out["data"]])
  
  if pbar:
    pbar.update(1)
  return _arr, None


def main(
  pdf: str,
  bucket_size = 16,
  collection_name = "my-test-collection"
):

  # open the pdf
  doc = fitz.open(pdf)

  # extract the text
  page_text = []
  print("Total pages:", doc.page_count)
  for pno in range(doc.page_count):
    page_text.append(doc.get_page_text(pno))

  # rules for the text:
  # 01 page contains atleast 10 words
  # 02 if page contains > 700 tokens ~ 2500 chars we chunk into two parts with overlap
  payloads = []
  for i,p in enumerate(page_text):
    if len(p.strip().split()) < 10:
      continue
    
    chunk_size = 2500
    if len(p) > 2500:
      for j,k in enumerate(range(0, len(p), int(chunk_size * 0.8))):
        payloads.append({
          "doc": pdf,
          "page_no": i,
          "chunk": j,
          "text": p[k:k+chunk_size]
        })
    else:
      payloads.append({
        "doc": pdf,
        "page_no": i,
        "chunk": 0,
        "text": p
      })

  print("Total payloads:", len(payloads))
  print("Sample payload:", payloads[0])
  print("Loading embeddings")
  pbar = trange(len(payloads))
  all_items = []
  buckets = [
    payloads[i:i+bucket_size]
    for i in range(0, len(payloads), bucket_size)
  ]
  for b in buckets:
    full_out = threaded_map(
      fn = get_embedding,
      inputs = [(x, pbar) for x in b],
      max_threads = 16
    )
    all_items.extend(full_out)

  embedding = np.vstack([x[0] for x in all_items])
  print(embedding.shape)

  # save the embeddings
  # https://qdrant.tech/documentation/tutorials/bulk-upload/#upload-directly-to-disk
  recreate_collection(collection_name, 1536) # OpenAI embedding dim
  disable_indexing(collection_name)

  client.upload_collection(
    collection_name = collection_name,
    vectors = embedding,
    payload = payloads,
    ids = None, # Vector ids will be assigned automatically
    batch_size = 256 # How many vectors will be uploaded in a single request?
  )

  enable_indexing(collection_name)

if __name__ == "__main__":
  # "./blitzscaling_slides.pdf"
  Fire(main)
