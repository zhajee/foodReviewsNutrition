from __future__ import print_function
import faiss as faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from openai import OpenAI
client = OpenAI(api_key="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")
# imports
import pandas as pd
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer

prompt_template = """
    <s> [INST] <<SYS>>
    Use the content provided to answer the question at the end. If you don't know the answer don't try to make up the answer.
    <</SYS>>

    Context:
    ----------
    {context}
    ----------

    Question: {question} [/INST]
    """
PROMPT = PromptTemplate(
     template=prompt_template,
     input_variables=["context","question"]
)

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset
input_datapath = "../data/finesmall-food.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
df["text"] = (
    df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
print(df.head(2))
#print(df["text"].head(2))
#print(df.Text.tolist())
#print(df.UserId)
print("Fine Food Reviews Summary")
print(df.Summary)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_data_chunks(data: str, chunk_size: int):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=5, separator="\n", length_function=len)
    chunks = text_splitter.split_text(data)
    return chunks

def create_knowledge_hub(chunks: list):
#    print("Printing embeddings chunks type", type(chunks.tolist()))
    embeddings = chunks.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    embeddings_list = embeddings.tolist()
    print("Type and Len of Embeddings List",type(embeddings), type(embeddings_list), len(embeddings_list))
#    print(embeddings_list[0])
    print("Len Chunks",len(chunks))
    text_embeddings = zip(chunks.tolist(), embeddings_list)
    text_embeddings_list = list(text_embeddings)
    print("Review print first Text and Corresponding Embedding Vector")
    print("===========================================================")
    print(text_embeddings_list[0])
    print("Done Creating text Embeddings")
    vtexts = [t[0] for t in text_embeddings_list]
    vembeddings = [t[1] for t in text_embeddings_list]
    print("len [Embedig Vector Len, Num-Embedding-Vectors]",[len(vembeddings[0]), len(vembeddings)])
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    print("OpenAI embed type\n",embeddings)
    knowledge_hub = FAISS.from_embeddings(text_embeddings_list, embeddings)
    return knowledge_hub


if __name__ == '__main__':
    sentenses = df.Text.apply(lambda x: x)
    sentenses = df.Text.tolist()
    print("Sentenses", type(sentenses))
    print(sentenses)
    model = SentenceTransformer('bert-base-nli-mean-tokens')
# create sentence embeddings
    sentence_embeddings = model.encode(sentenses)
    print(sentence_embeddings.shape)
    print("Type(sentence_embedding) and shape", type(sentence_embeddings), sentence_embeddings.shape[1])
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    faiss.normalize_L2(sentence_embeddings)
    index.is_trained
    index.train(sentence_embeddings)
    print(index.ntotal)
    index.add(sentence_embeddings)
    question="Healthy dog food"
    k = 4
#    xq = model.encode(["Healthy dog food"])
    xq = model.encode([question])

    D, I = index.search(xq, k)  # search
    matches=([f'{i}: {sentenses[i]}' for i in I[0]])
    print(f'Q: {question}\n')
    print(f'Matches (in order Similarity Embedding Ranking)\n')
    for result in matches:
        print(result);
"""
    print(I,type(D), type(I))
    for i in I:
        print(sentence_embeddings[i])
        print(sentenses[i])
    print(D)
"""

