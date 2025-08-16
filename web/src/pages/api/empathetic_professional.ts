import type { NextApiRequest, NextApiResponse } from 'next'

// Prefer IPv4 loopback to avoid potential IPv6 (::1) resolution issues on Windows
const RAW_API_BASE_URL = process.env.API_BASE_URL || 'http://127.0.0.1:8000'
const API_BASE_URL = RAW_API_BASE_URL.replace('localhost', '127.0.0.1')
const DEFAULT_TIMEOUT_MS = Number(process.env.API_TIMEOUT_MS || 65000)
const USE_FASTAPI_V2 = (process.env.USE_FASTAPI_V2 || 'true').toLowerCase() === 'true'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { question, type, history, session_id, weekly_goal, feasibility, anxiety_level, user_profile } = req.body

    // Optional timeout override via query
    const timeoutOverride = req.query.timeoutMs ? Number(req.query.timeoutMs) : undefined
    const API_TIMEOUT_MS = Number.isFinite(timeoutOverride as number) && (timeoutOverride as number) > 0
      ? (timeoutOverride as number)
      : DEFAULT_TIMEOUT_MS

    // Add input validation
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Question is required and must be a string' })
    }

    console.log('Next.js API (Empathetic Professional): Processing request:', { 
      question: question.substring(0, 50), 
      type, 
      historyLength: history?.length || 0,
      timeoutMs: API_TIMEOUT_MS,
      v2: USE_FASTAPI_V2,
    })

    try {
      console.log('Next.js API: Calling FastAPI empathetic_professional endpoint')

      // Add timeout to avoid hanging requests
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS)

      const path = USE_FASTAPI_V2 ? '/api/v2/empathetic_professional' : '/api/empathetic_professional'
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question, 
          type: type || 'empathetic_professional', 
          history: history || [],
          session_id,
          weekly_goal,
          feasibility,
          anxiety_level,
          user_profile,
        }),
        signal: controller.signal,
        cache: 'no-store',
      });

      clearTimeout(timer)

      console.log('Next.js API: FastAPI response status:', response.status)

      let data;
      try {
        const responseText = await response.text();
        console.log('Next.js API: Raw response text length:', responseText.length);
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Next.js API: JSON parse error:', parseError);
        return res.status(500).json({ error: 'Failed to parse backend response' })
      }
      
      if (!response.ok) {
        console.error(`Next.js API: Backend error: ${response.status} -`, data);
        return res.status(response.status).json({ error: data.detail || data.error || data || 'Backend error' })
      }

      console.log('Next.js API: Parsed data:', { 
        hasAnswer: !!data.answer, 
        answerLength: data.answer?.length || 0,
        status: data.status,
        tone: data.tone
      })

      // FastAPI returns format: { answer, question, tone, status }
      return res.status(200).json({ 
        answer: data.answer || 'No response received',
        question: data.question,
        tone: data.tone,
        status: data.status
      })
      
    } catch (fetchError) {
      console.error('Next.js API: Fetch error:', fetchError);
      const errorMessage = fetchError instanceof Error ? fetchError.message : 'Unknown network error';
      return res.status(500).json({ error: `Network error: ${errorMessage}` })
    }
  } catch (error) {
    console.error('Next.js API: General error:', error)
    return res.status(500).json({ error: 'Internal server error' })
  }
} 