import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { RegisterCredentials } from '@/types';
import { useAuth } from '@/lib/AuthContext';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { Eye, EyeOff, Mail, Lock, User, AlertCircle, Check } from 'lucide-react';

interface RegisterFormProps {
  onSwitchToLogin: () => void;
  onSuccess?: () => void;
  className?: string;
}

const RegisterForm: React.FC<RegisterFormProps> = ({
  onSwitchToLogin,
  onSuccess,
  className
}) => {
  const { register: registerUser, isLoading, error, clearError } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
    reset
  } = useForm<RegisterCredentials>();

  const password = watch('password');

  const onSubmit = async (data: RegisterCredentials) => {
    try {
      setFormError(null);
      clearError();
      await registerUser(data);
      reset();
      onSuccess?.();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Registration failed';
      setFormError(errorMessage);
    }
  };

  const displayError = formError || error;

  return (
    <div className={`w-full max-w-md mx-auto ${className}`}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-secondary-900 mb-2">
          Create Account
        </h2>
        <p className="text-secondary-600">
          Join us to get started with your journey
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
          <label htmlFor="name" className="block text-sm font-medium text-secondary-700">
            Full Name
          </label>
          <div className="relative">
            <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <Input
              id="name"
              type="text"
              placeholder="Enter your full name"
              className="pl-10"
              error={errors.name?.message}
              {...register('name', {
                required: 'Full name is required',
                minLength: {
                  value: 2,
                  message: 'Name must be at least 2 characters'
                }
              })}
            />
          </div>
          {errors.name && (
            <p className="text-sm text-error-600">{errors.name.message}</p>
          )}
        </div>

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
              placeholder="Create a password"
              className="pl-10 pr-10"
              error={errors.password?.message}
              {...register('password', {
                required: 'Password is required',
                minLength: {
                  value: 8,
                  message: 'Password must be at least 8 characters'
                },
                pattern: {
                  value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
                  message: 'Password must contain at least one uppercase letter, one lowercase letter, and one number'
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

        <div className="space-y-2">
          <label htmlFor="confirmPassword" className="block text-sm font-medium text-secondary-700">
            Confirm Password
          </label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <Input
              id="confirmPassword"
              type={showConfirmPassword ? 'text' : 'password'}
              placeholder="Confirm your password"
              className="pl-10 pr-10"
              error={errors.confirmPassword?.message}
              {...register('confirmPassword', {
                required: 'Please confirm your password',
                validate: (value) => value === password || 'Passwords do not match'
              })}
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-secondary-400 hover:text-secondary-600"
            >
              {showConfirmPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          {errors.confirmPassword && (
            <p className="text-sm text-error-600">{errors.confirmPassword.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <label className="flex items-start">
            <input
              type="checkbox"
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded mt-1"
              {...register('agreeToTerms', {
                required: 'You must agree to the terms and conditions'
              })}
            />
            <span className="ml-2 text-sm text-secondary-600">
              I agree to the{' '}
              <button
                type="button"
                className="text-primary-600 hover:text-primary-500 font-medium"
              >
                Terms of Service
              </button>{' '}
              and{' '}
              <button
                type="button"
                className="text-primary-600 hover:text-primary-500 font-medium"
              >
                Privacy Policy
              </button>
            </span>
          </label>
          {errors.agreeToTerms && (
            <p className="text-sm text-error-600">{errors.agreeToTerms.message}</p>
          )}
        </div>

        <Button
          type="submit"
          variant="primary"
          size="lg"
          loading={isLoading}
          disabled={isLoading}
          className="w-full"
        >
          {isLoading ? 'Creating Account...' : 'Create Account'}
        </Button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-secondary-600">
          Already have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToLogin}
            className="text-primary-600 hover:text-primary-500 font-medium"
          >
            Sign in
          </button>
        </p>
      </div>
    </div>
  );
};

export default RegisterForm; 