#  Intelligent Fusion System for Mental Health

This is an intelligent conversational AI designed for mental health support, powered by a multi-modal architecture that integrates Retrieval-Augmented Generation (RAG) with advanced Prompt Engineering.

The system fuses Cognitive Behavioral Therapy (CBT) therapeutic techniques with ICD-11 diagnostic standards to deliver a safe, personalized, and empathetic user experience. The backend is built with FastAPI for modular orchestration, while the frontend uses Next.js to create an adaptive and responsive chat interface.

---

##  Core Features

- **Multi-Tone Emotional Intelligence** — Dynamically adjusts its communication style between caring, professional, and balanced tones to provide empathetic interactions.
- **Self-Optimizing Prompts** — Features built-in repair and judge services that continuously refine prompt outputs for better response quality over time.
- **RAG-Enhanced Memory** — Utilizes a vector store to retrieve context, ensuring seamless continuity and accuracy across multi-turn conversations.
- **CBT-Oriented Dialogue Flow** — Follows structured conversation stages inspired by Cognitive Behavioral Therapy (Assessment → Adjustment → Reflection → Wrap-up).
- **Safety-Embedded Framework** — All conversational routes are pre-validated by structured contracts and safety filters to ensure user well-being.

---

##  Tech Stack

- **Backend:** FastAPI, Python 3.10+  
- **Frontend:** Next.js, TypeScript, TailwindCSS  
- **LLM Integration:** Supports local or remote LLaMA-based adapters  
- **Storage:** Vector Store (FAISS / Chroma)  
- **Prompt Registry:** YAML-based structured prompt definitions  

---

##  Project Structure Overview

- /app/
    - api/ - FastAPI endpoints
    - clients/ - LLM and vector store clients
    - orchestration/ - Core flow, routing, and judging logic
    - prompts/ - YAML-based prompt registry (tones, flows, tasks)
    - services/ - RAG, chat, and memory services
    - utils/ - Helpers for validation, crisis detection, etc.
- /web/ - Next.js frontend application
