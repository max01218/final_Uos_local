# ICD-11 RAG Web Frontend (Next.js)

## Directory Structure

```
web/
├── README.md
├── package.json
├── next.config.js
├── public/
├── src/
│   ├── pages/
│   │   ├── index.tsx         # Main page
│   │   └── api/
│   │       └── rag.ts        # API route, forwards to backend RAG
│   ├── components/
│   │   ├── PromptPanel.tsx   # Left input area
│   │   ├── RAGPanel.tsx      # Top knowledge source area
│   │   ├── FMPanel.tsx       # Central main processing area
│   │   ├── DialoguePanel.tsx # Right dialogue area
│   │   └── DiagnosisPanel.tsx# Bottom diagnosis area
│   └── styles/
│       └── globals.css
```

## Start Guide

```bash
cd web
npm install
npm run dev
```

## Description
- This frontend uses Next.js 14+ with TypeScript support
- Clear component layout for easy future feature expansion
- `/api/rag.ts` can interface with backend Python RAG system 