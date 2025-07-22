# 重新執行 FAISS index 建立流程，這次在 prompts 資料夾中建立向量庫
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pathlib import Path
# 只選取 prompts 資料夾下的 .txt 檔案
prompts_dir = Path("prompts")  # ✅ 把字串轉成 Path 物件
text_files = list(prompts_dir.rglob("*.txt"))
documents = []

for file in text_files:
    try:
        content = file.read_text(encoding="utf-8").strip()
        if content:
            documents.append(Document(page_content=content))
    except Exception:
        continue  # 忽略讀取錯誤的檔案

# 初始化句向量嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 建立 FAISS 向量庫
vectorstore = FAISS.from_documents(documents, embedding_model)

# 儲存為 FastAPI 可用的 embeddings 索引資料夾
output_path = Path("embeddings")
vectorstore.save_local(str(output_path))

# 顯示儲存結果
print(list(output_path.iterdir()))

