import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/lib/AuthContext';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  fallback?: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  redirectTo = '/auth',
  fallback
}) => {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push(redirectTo);
    }
  }, [isAuthenticated, isLoading, router, redirectTo]);

  // Show loading state while checking authentication
  if (isLoading) {
    return fallback || (
      <div className="min-h-screen flex items-center justify-center bg-secondary-50">
        <div className="text-center">
          <Loader2 className="h-8 w-8 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-secondary-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Show children only if authenticated
  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
};

export default ProtectedRoute; 