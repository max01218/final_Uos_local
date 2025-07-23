# Re-execute FAISS index creation process, building vector database in prompts folder
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pathlib import Path
# Select only .txt files under prompts folder
prompts_dir = Path("prompts")  # Convert string to Path object
text_files = list(prompts_dir.rglob("*.txt"))
documents = []

for file in text_files:
    try:
        content = file.read_text(encoding="utf-8").strip()
        if content:
            documents.append(Document(page_content=content))
    except Exception:
        continue  # Skip files with read errors

# Initialize sentence embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save as embeddings index folder for FastAPI usage
output_path = Path("embeddings")
vectorstore.save_local(str(output_path))

# Display save results
print(list(output_path.iterdir()))

