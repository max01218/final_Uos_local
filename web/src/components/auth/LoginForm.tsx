import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { LoginCredentials } from '@/types';
import { useAuth } from '@/lib/AuthContext';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { Eye, EyeOff, Mail, Lock, AlertCircle } from 'lucide-react';

interface LoginFormProps {
  onSwitchToRegister: () => void;
  onSuccess?: () => void;
  className?: string;
}

const LoginForm: React.FC<LoginFormProps> = ({
  onSwitchToRegister,
  onSuccess,
  className
}) => {
  const { login, isLoading, error, clearError } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset
  } = useForm<LoginCredentials>();

  const onSubmit = async (data: LoginCredentials) => {
    try {
      setFormError(null);
      clearError();
      await login(data);
      reset();
      onSuccess?.();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed';
      setFormError(errorMessage);
    }
  };

  const displayError = formError || error;

  return (
    <div className={`w-full max-w-md mx-auto ${className}`}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-secondary-900 mb-2">
          Welcome Back
        </h2>
        <p className="text-secondary-600">
          Sign in to your account to continue
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {displayError && (
          <div className="flex items-center gap-2 p-3 bg-error-50 border border-error-200 rounded-lg">
            <AlertCircle className="h-4 w-4 text-error-500 flex-shrink-0" />
            <span className="text-sm text-error-700">{displayError}</span>
          </div>
        )}

        <div className="space-y-2">
          <label htmlFor="email" className="block text-sm font-medium text-secondary-700">
            Email Address
          </label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              className="pl-10"
              error={errors.email?.message}
              {...register('email', {
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Invalid email address'
                }
              })}
            />
          </div>
          {errors.email && (
            <p className="text-sm text-error-600">{errors.email.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <label htmlFor="password" className="block text-sm font-medium text-secondary-700">
            Password
          </label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <Input
              id="password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter your password"
              className="pl-10 pr-10"
              error={errors.password?.message}
              {...register('password', {
                required: 'Password is required',
                minLength: {
                  value: 8,
                  message: 'Password must be at least 8 characters'
                }
              })}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-secondary-400 hover:text-secondary-600"
            >
              {showPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          {errors.password && (
            <p className="text-sm text-error-600">{errors.password.message}</p>
          )}
        </div>

        <div className="flex items-center justify-between">
          <label className="flex items-center">
            <input
              type="checkbox"
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              {...register('rememberMe')}
            />
            <span className="ml-2 text-sm text-secondary-600">
              Remember me
            </span>
          </label>
          <button
            type="button"
            className="text-sm text-primary-600 hover:text-primary-500 font-medium"
          >
            Forgot password?
          </button>
        </div>

        <Button
          type="submit"
          variant="primary"
          size="lg"
          loading={isLoading}
          disabled={isLoading}
          className="w-full"
        >
          {isLoading ? 'Signing in...' : 'Sign In'}
        </Button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-secondary-600">
          Don't have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToRegister}
            className="text-primary-600 hover:text-primary-500 font-medium"
          >
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
};

export default LoginForm; 