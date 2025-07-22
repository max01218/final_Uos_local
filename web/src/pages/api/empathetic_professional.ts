import type { NextApiRequest, NextApiResponse } from 'next'

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { question, type, history } = req.body

    // Add input validation
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Question is required and must be a string' })
    }

    console.log('Next.js API (Empathetic Professional): Processing request:', { 
      question: question.substring(0, 50), 
      type, 
      historyLength: history?.length || 0 
    })

    try {
      console.log('Next.js API: Calling FastAPI empathetic_professional endpoint')
      
      const response = await fetch(`${API_BASE_URL}/api/empathetic_professional`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question, 
          type: type || 'empathetic_professional', 
          history: history || [] 
        }),
      });

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
      return res.status(500).json({ error: `Network error: ${fetchError.message}` })
    }
  } catch (error) {
    console.error('Next.js API: General error:', error)
    return res.status(500).json({ error: 'Internal server error' })
  }
} 