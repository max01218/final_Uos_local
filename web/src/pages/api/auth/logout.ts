import { NextApiRequest, NextApiResponse } from 'next';
import { ApiResponse } from '@/types';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ApiResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({
      success: false,
      error: 'Method not allowed',
      timestamp: new Date()
    });
  }

  try {
    // In production, you would:
    // 1. Invalidate the token in your database
    // 2. Clear any server-side sessions
    // 3. Add the token to a blacklist

    res.status(200).json({
      success: true,
      message: 'Logout successful',
      timestamp: new Date()
    });

  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      timestamp: new Date()
    });
  }
} 