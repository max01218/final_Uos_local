import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { InputProps } from '@/types';

const Input = forwardRef<HTMLInputElement, InputProps>(({
  type = 'text',
  placeholder,
  value,
  onChange,
  onBlur,
  onFocus,
  disabled = false,
  required = false,
  error,
  success = false,
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedby,
  ...props
}, ref) => {
  const baseClasses = 'w-full rounded-lg border bg-white px-4 py-3 text-sm placeholder-secondary-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-0';
  
  const stateClasses = {
    default: 'border-secondary-300 focus:border-primary-500 focus:ring-primary-500/20',
    error: 'border-error-300 focus:border-error-500 focus:ring-error-500/20',
    success: 'border-success-300 focus:border-success-500 focus:ring-success-500/20',
  };
  
  const getStateClass = () => {
    if (error) return stateClasses.error;
    if (success) return stateClasses.success;
    return stateClasses.default;
  };
  
  const classes = cn(
    baseClasses,
    getStateClass(),
    disabled && 'opacity-50 cursor-not-allowed bg-secondary-50',
    className
  );

  return (
    <div className="relative">
      <input
        ref={ref}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => onChange?.(e)}
        onBlur={onBlur}
        onFocus={onFocus}
        disabled={disabled}
        required={required}
        aria-label={ariaLabel}
        aria-describedby={ariaDescribedby}
        aria-invalid={!!error}
        className={classes}
        {...props}
      />
      
      {error && (
        <p className="mt-1 text-xs text-error-600">
          {error}
        </p>
      )}
      
      {success && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-success-500">
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </div>
      )}
    </div>
  );
});

Input.displayName = 'Input';

export default Input; 