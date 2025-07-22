# backend/rag_chain.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# load FAISS index ---------------------------------------------------
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
store = FAISS.load_local("embeddings", embedder,
                         allow_dangerous_deserialization=True)
retriever = store.as_retriever(search_kwargs={"k": 3})

# load LLM -----------------------------------------------------------
tok = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat",
                                    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat",
    trust_remote_code=True,
    device_map="auto"             
)

# Create HuggingFace pipeline with better parameters
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tok.eos_token_id
)

# Wrap pipeline in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Improved prompt template for Qwen model
PROMPT = PromptTemplate(
    template=(
        "<|im_start|>system\n"
        "You are a professional mental-health assistant. Answer questions based on the provided context only.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    input_variables=["context", "question"],
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Test the system
print("Testing RAG system...")
result = rag_chain.invoke({"query": "What is depression?"})

print("\n" + "="*60)
print("RAG SYSTEM RESULT")
print("="*60)
print("Question: What is depression?")
print("\nAnswer:", result['result'])
print("\nSource Documents:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"{i}. Code: {doc.metadata.get('code', 'Unknown')}")
    print(f"   Content preview: {doc.page_content[:100]}...")
print("="*60)
