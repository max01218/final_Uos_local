import { 
  registerWithFirebase, 
  loginWithFirebase, 
  logoutFromFirebase,
  onAuthStateChange
} from './firebase';
import { LoginCredentials, RegisterCredentials, AuthUser } from '@/types';

// Token management
export const getStoredToken = (): string | null => {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('auth_token');
};

export const setStoredToken = (token: string): void => {
  if (typeof window === 'undefined') return;
  localStorage.setItem('auth_token', token);
};

export const removeStoredToken = (): void => {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('auth_token');
};

export const getStoredRefreshToken = (): string | null => {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('refresh_token');
};

export const setStoredRefreshToken = (token: string): void => {
  if (typeof window === 'undefined') return;
  localStorage.setItem('refresh_token', token);
};

export const removeStoredRefreshToken = (): void => {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('refresh_token');
};

// User data management
export const getStoredUser = (): AuthUser | null => {
  if (typeof window === 'undefined') return null;
  const userData = localStorage.getItem('user_data');
  if (!userData) return null;
  
  try {
    const user = JSON.parse(userData);
    return {
      ...user,
      createdAt: new Date(user.createdAt),
      lastActive: new Date(user.lastActive)
    };
  } catch (error) {
    console.error('Error parsing stored user data:', error);
    return null;
  }
};

export const setStoredUser = (user: AuthUser): void => {
  if (typeof window === 'undefined') return;
  localStorage.setItem('user_data', JSON.stringify(user));
};

export const removeStoredUser = (): void => {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('user_data');
};

// Token validation
export const isTokenValid = (token: string): boolean => {
  if (!token) return false;
  
  try {
    // Decode JWT token to check expiration
    const payload = JSON.parse(atob(token.split('.')[1]));
    const currentTime = Date.now() / 1000;
    
    return payload.exp > currentTime;
  } catch (error) {
    console.error('Error validating token:', error);
    return false;
  }
};

// API functions using Firebase
export const registerUser = async (credentials: RegisterCredentials) => {
  try {
    const response = await registerWithFirebase(credentials);
    return response;
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

export const loginUser = async (credentials: LoginCredentials) => {
  try {
    const response = await loginWithFirebase(credentials);
    return response;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

export const logoutUser = async () => {
  try {
    await logoutFromFirebase();
    
    // Clear local storage
    removeStoredToken();
    removeStoredRefreshToken();
    removeStoredUser();
  } catch (error) {
    console.error('Logout error:', error);
    throw error;
  }
};

// Auth state listener
export const setupAuthStateListener = (callback: (user: AuthUser | null) => void) => {
  return onAuthStateChange(callback);
}; 