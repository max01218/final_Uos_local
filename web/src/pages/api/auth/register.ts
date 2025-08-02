import { NextApiRequest, NextApiResponse } from 'next';
import { RegisterCredentials, ApiResponse, AuthResponse } from '@/types';

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
    const { name, email, password, confirmPassword, agreeToTerms }: RegisterCredentials = req.body;

    // Validation
    if (!name || !email || !password || !confirmPassword) {
      return res.status(400).json({
        success: false,
        error: 'All fields are required',
        timestamp: new Date()
      });
    }

    if (password !== confirmPassword) {
      return res.status(400).json({
        success: false,
        error: 'Passwords do not match',
        timestamp: new Date()
      });
    }

    if (password.length < 8) {
      return res.status(400).json({
        success: false,
        error: 'Password must be at least 8 characters long',
        timestamp: new Date()
      });
    }

    if (!agreeToTerms) {
      return res.status(400).json({
        success: false,
        error: 'You must agree to the terms and conditions',
        timestamp: new Date()
      });
    }

    // Check if user already exists
    const existingUser = users.find(user => user.email === email);
    if (existingUser) {
      return res.status(409).json({
        success: false,
        error: 'User with this email already exists',
        timestamp: new Date()
      });
    }

    // Create new user
    const newUser = {
      id: Date.now().toString(),
      email,
      name,
      password: await hashPassword(password), // In production, use bcrypt
      createdAt: new Date(),
      lastActive: new Date(),
      isVerified: false,
      preferences: {
        theme: 'light' as const,
        language: 'zh-CN' as const,
        notifications: true,
        autoSave: true,
        defaultTone: 'professional' as const
      }
    };

    users.push(newUser);

    // Generate tokens (in production, use JWT)
    const token = generateToken(newUser.id);
    const refreshToken = generateRefreshToken(newUser.id);

    const authResponse: AuthResponse = {
      user: {
        id: newUser.id,
        email: newUser.email,
        name: newUser.name,
        createdAt: newUser.createdAt,
        lastActive: newUser.lastActive,
        isVerified: newUser.isVerified,
        preferences: newUser.preferences
      },
      token,
      refreshToken,
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
    };

    res.status(201).json({
      success: true,
      data: authResponse,
      message: 'User registered successfully',
      timestamp: new Date()
    });

  } catch (error) {
    console.error('Registration error:', error);
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