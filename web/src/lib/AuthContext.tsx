import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { AuthState, AuthUser, LoginCredentials, RegisterCredentials } from '@/types';
import { 
  getStoredToken, 
  setStoredToken, 
  removeStoredToken,
  getStoredRefreshToken,
  setStoredRefreshToken,
  removeStoredRefreshToken,
  getStoredUser,
  setStoredUser,
  removeStoredUser,
  loginUser,
  registerUser,
  logoutUser,
  isTokenValid,
  setupAuthStateListener
} from './auth';

// Action types
type AuthAction =
  | { type: 'LOGIN_START' }
  | { type: 'LOGIN_SUCCESS'; payload: { user: AuthUser; token: string; refreshToken: string } }
  | { type: 'LOGIN_FAILURE'; payload: string }
  | { type: 'LOGOUT' }
  | { type: 'CLEAR_ERROR' }
  | { type: 'UPDATE_USER'; payload: AuthUser }
  | { type: 'SET_LOADING'; payload: boolean };

// Initial state
const initialState: AuthState = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
};

// Reducer
function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'LOGIN_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      };
    case 'LOGIN_FAILURE':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: action.payload,
      };
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      };
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };
    case 'UPDATE_USER':
      return {
        ...state,
        user: action.payload,
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload,
      };
    default:
      return state;
  }
}

// Context
interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (credentials: RegisterCredentials) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
  updateUser: (user: AuthUser) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Provider component
interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Initialize auth state from Firebase
  useEffect(() => {
    const unsubscribe = setupAuthStateListener((user) => {
      if (user) {
        // User is signed in
        const token = getStoredToken();
        const refreshToken = getStoredRefreshToken();
        
        if (token && isTokenValid(token)) {
          dispatch({
            type: 'LOGIN_SUCCESS',
            payload: { user, token, refreshToken: refreshToken || '' }
          });
        } else {
          // Token is invalid, clear everything
          removeStoredToken();
          removeStoredRefreshToken();
          removeStoredUser();
          dispatch({ type: 'LOGOUT' });
        }
      } else {
        // User is signed out
        removeStoredToken();
        removeStoredRefreshToken();
        removeStoredUser();
        dispatch({ type: 'LOGOUT' });
      }
    });

    return () => unsubscribe();
  }, []);

  // Login function
  const login = async (credentials: LoginCredentials) => {
    try {
      dispatch({ type: 'LOGIN_START' });
      
      const response = await loginUser(credentials);
      
      // Store tokens and user data
      setStoredToken(response.token);
      setStoredRefreshToken(response.refreshToken);
      setStoredUser(response.user);
      
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: response.user,
          token: response.token,
          refreshToken: response.refreshToken
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed';
      dispatch({ type: 'LOGIN_FAILURE', payload: errorMessage });
      throw error;
    }
  };

  // Register function
  const register = async (credentials: RegisterCredentials) => {
    try {
      dispatch({ type: 'LOGIN_START' });
      
      const response = await registerUser(credentials);
      
      // Store tokens and user data
      setStoredToken(response.token);
      setStoredRefreshToken(response.refreshToken);
      setStoredUser(response.user);
      
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: response.user,
          token: response.token,
          refreshToken: response.refreshToken
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Registration failed';
      dispatch({ type: 'LOGIN_FAILURE', payload: errorMessage });
      throw error;
    }
  };

  // Logout function
  const logout = async () => {
    try {
      await logoutUser();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      dispatch({ type: 'LOGOUT' });
    }
  };

  // Clear error
  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  // Update user
  const updateUser = (user: AuthUser) => {
    setStoredUser(user);
    dispatch({ type: 'UPDATE_USER', payload: user });
  };

  const value: AuthContextType = {
    ...state,
    login,
    register,
    logout,
    clearError,
    updateUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
} 