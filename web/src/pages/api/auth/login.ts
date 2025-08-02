import { NextApiRequest, NextApiResponse } from 'next';
import { LoginCredentials, ApiResponse, AuthResponse } from '@/types';

// Mock database - in production, use a real database
const users: any[] = [];

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ApiResponse<AuthResponse>>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({
      success: false,
      error: 'Method not allowed',
      timestamp: new Date()
    });
  }

  try {
    const { email, password, rememberMe }: LoginCredentials = req.body;

    // Validation
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        error: 'Email and password are required',
        timestamp: new Date()
      });
    }

    // Find user by email
    const user = users.find(u => u.email === email);
    if (!user) {
      return res.status(401).json({
        success: false,
        error: 'Invalid email or password',
        timestamp: new Date()
      });
    }

    // Verify password
    const hashedPassword = await hashPassword(password);
    if (user.password !== hashedPassword) {
      return res.status(401).json({
        success: false,
        error: 'Invalid email or password',
        timestamp: new Date()
      });
    }

    // Update last active
    user.lastActive = new Date();

    // Generate tokens
    const token = generateToken(user.id);
    const refreshToken = generateRefreshToken(user.id);
    const expiresAt = rememberMe 
      ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days
      : new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

    const authResponse: AuthResponse = {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        createdAt: user.createdAt,
        lastActive: user.lastActive,
        isVerified: user.isVerified,
        preferences: user.preferences
      },
      token,
      refreshToken,
      expiresAt
    };

    res.status(200).json({
      success: true,
      data: authResponse,
      message: 'Login successful',
      timestamp: new Date()
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      timestamp: new Date()
    });
  }
}

// Mock functions - replace with real implementations
async function hashPassword(password: string): Promise<string> {
  // In production, use bcrypt or similar
  return Buffer.from(password).toString('base64');
}

function generateToken(userId: string): string {
  // In production, use JWT
  return Buffer.from(`${userId}-${Date.now()}`).toString('base64');
}

function generateRefreshToken(userId: string): string {
  // In production, use JWT with longer expiry
  return Buffer.from(`${userId}-refresh-${Date.now()}`).toString('base64');
} 