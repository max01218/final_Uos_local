// Message related types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  type?: MessageType;
  metadata?: MessageMetadata;
}

export type MessageType = 'normal' | 'safety_alert' | 'suggestion' | 'follow_up' | 'error';

export interface MessageMetadata {
  confidence?: number;
  fusion_strategy?: string;
  safety_notes?: string[];
  safety_alerts?: SafetyAlert[];
  emotion_analysis?: EmotionAnalysis;
  follow_up_suggestions?: string[];
  source_breakdown?: Record<string, number>;
  emotion_detected?: string;
  urgency_level?: number;
  processing_time?: number;
}

// Conversation related types
export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  tone: ToneType;
  metadata?: ConversationMetadata;
  userId?: string; // Add user association
  isArchived?: boolean; // Add archive status
}

export interface ConversationMetadata {
  totalMessages: number;
  averageResponseTime: number;
  safetyAlerts: number;
  userEmotions: string[];
}

// Tone types
export type ToneType = 'professional' | 'caring' | 'empathetic_professional';

export interface ToneConfig {
  id: ToneType;
  label: string;
  description: string;
  icon: string;
  color: string;
}

// Authentication related types
export interface AuthUser {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
  lastActive: Date;
  isVerified: boolean;
  preferences: UserPreferences;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterCredentials {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  agreeToTerms: boolean;
}

export interface AuthResponse {
  user: AuthUser;
  token: string;
  refreshToken: string;
  expiresAt: Date;
}

export interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  newPassword: string;
  confirmPassword: string;
}

export interface EmailVerification {
  token: string;
}

// User related types
export interface User {
  id: string;
  name?: string;
  email?: string;
  preferences: UserPreferences;
  createdAt: Date;
  lastActive: Date;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: 'zh-CN' | 'en-US';
  notifications: boolean;
  autoSave: boolean;
  defaultTone: ToneType;
}

// Feedback related types
export interface Feedback {
  id: string;
  messageId: string;
  type: 'positive' | 'negative' | 'neutral';
  rating?: number;
  comment?: string;
  timestamp: Date;
  metadata?: FeedbackMetadata;
}

export interface FeedbackMetadata {
  helpful: boolean;
  accuracy: number;
  empathy: number;
  clarity: number;
}

// Safety related types
export interface SafetyAlert {
  level: 'low' | 'medium' | 'high' | 'critical';
  type: 'self_harm' | 'violence' | 'crisis' | 'other';
  message: string;
  recommendations: string[];
  resources: SafetyResource[];
}

export interface SafetyResource {
  name: string;
  description: string;
  url: string;
  phone?: string;
  available: boolean;
}

// Emotion detection types
export interface EmotionAnalysis {
  primary: string;
  confidence: number;
  secondary?: string[];
  intensity: number;
  suggestions: string[];
}

// API response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
}

export interface ChatApiRequest {
  question: string;
  type: ToneType;
  tone: ToneType;
  history: Message[];
  metadata?: {
    userAgent?: string;
    sessionId?: string;
    timestamp?: number;
  };
}

export interface ChatApiResponse {
  answer: string;
  confidence?: number;
  safety_alerts?: SafetyAlert[];
  emotion_analysis?: EmotionAnalysis;
  follow_up_suggestions?: string[];
  processing_time?: number;
}

// Component Props types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
  'aria-label'?: string;
  'aria-describedby'?: string;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface InputProps extends BaseComponentProps {
  id?: string;
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url';
  placeholder?: string;
  value?: string;
  onChange?: any; // Allow any onChange function to support React Hook Form
  onBlur?: any; // Allow any onBlur function to support React Hook Form
  onFocus?: () => void;
  disabled?: boolean;
  required?: boolean;
  error?: string;
  success?: boolean;
  min?: string | number;
  max?: string | number;
  step?: string | number;
  pattern?: string;
  autoComplete?: string;
  autoFocus?: boolean;
  readOnly?: boolean;
  name?: string;
}

// Theme types
export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    success: string;
    warning: string;
  };
}

// Performance monitoring types
export interface PerformanceMetrics {
  pageLoadTime: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
  timeToInteractive: number;
}

// Error types
export interface AppError {
  id: string;
  type: 'network' | 'validation' | 'auth' | 'server' | 'unknown';
  message: string;
  details?: any;
  timestamp: Date;
  userAgent?: string;
  stack?: string;
}

// Configuration types
export interface AppConfig {
  api: {
    baseUrl: string;
    timeout: number;
    retries: number;
  };
  features: {
    emotionDetection: boolean;
    safetyAlerts: boolean;
    feedback: boolean;
    analytics: boolean;
  };
  limits: {
    maxMessageLength: number;
    maxHistoryLength: number;
    rateLimit: number;
  };
} 