# Toward an AI Mental Therapist with Large Language Models

This repository contains my MSc Computer Science dissertation project at the University of Southampton. I designed and implemented a web-based conversational system that explores how large language models can provide more structured and consistent mental-health support through retrieval, prompt optimisation and safety-aware dialogue control.

The project is intended as a research prototype. It does not provide medical diagnosis and is not a replacement for qualified mental-health professionals.

## Project overview

General-purpose language models can respond fluently, but fluency alone is not enough for a sensitive conversation. My focus was therefore on the system around the model: how a request is routed, what reference material is retrieved, how tone is controlled, and how unsafe or unsuitable responses are detected before they reach the user.

The application combines a FastAPI backend with a Next.js interface. The main dialogue flow follows four stages:

1. **Router** – classifies the request as greeting, mental-health support, information or another type of conversation.
2. **Generator** – produces a response using the selected tone, conversation context and retrieved material.
3. **Judge and Repair** – checks structure, relevance and safety, then rewrites the answer when required.
4. **Naturalizer** – improves the final wording so that the response feels less mechanical.

## What I implemented

- A multi-turn conversation flow with session history and working memory.
- Retrieval-Augmented Generation using material based on ICD-11 and public guidance from organisations such as NICE, NHS and WHO.
- Three response styles: Professional, Caring and Balanced.
- A prompt-first optimisation workflow based on OPRO, used to compare and improve prompt candidates.
- Crisis-routing and post-generation checks for high-risk language, diagnosis claims and medication advice.
- A FastAPI service layer and a responsive Next.js/TypeScript chat interface.
- Structured message types for standard, therapeutic, safety and crisis responses.

During evaluation, the prompt optimisation workflow improved the combined scoring result from **2.375 to 5.375** across clarity, empathy, professionalism and effectiveness. The project also gave me practical experience balancing model quality, response latency and limited GPU memory across local, Colab and university HPC environments.

## Technology

- **Backend:** Python, FastAPI, Pydantic
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Models:** Qwen2.5 Instruct models and Llama-based prompt optimisation experiments
- **Retrieval:** FAISS and sentence-transformer embeddings
- **Prompt management:** YAML-based prompt registry
- **Development environments:** Windows, Google Colab and Apptainer-based HPC

## Repository layout

| Path | Purpose |
|---|---|
| `app/api/` | API routes exposed to the frontend |
| `app/orchestration/` | Routing, generation, judging and repair flow |
| `app/services/` | Conversation, memory and retrieval services |
| `app/clients/` | Model and vector-store integrations |
| `app/prompts/` | Prompt templates, tones and dialogue policies |
| `app/utils/` | Validation and safety-related utilities |
| `web/` | Next.js frontend |
| `OPRO_Streamlined/` | Prompt optimisation experiments |

## Frontend setup

```bash
cd web
npm install
npm run dev
```

Environment-specific values are documented in `.env.example`. Model paths and backend dependencies depend on the selected inference environment, so they must be configured before starting the FastAPI service.

## Research limitations

The system was evaluated as a dissertation prototype rather than a clinical product. Its safety checks can reduce obvious risks but cannot guarantee an appropriate response in every situation. Any real-world use would require professional clinical review, privacy controls, security testing and a clearly defined escalation process.
