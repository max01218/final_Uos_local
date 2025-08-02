import React, { useState } from 'react';
import { useRouter } from 'next/router';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import HomeButton from '@/components/ui/HomeButton';
import { Shield, Heart, Brain } from 'lucide-react';

interface AuthPageProps {
  defaultMode?: 'login' | 'register';
  redirectTo?: string;
  className?: string;
}

const AuthPage: React.FC<AuthPageProps> = ({
  defaultMode = 'login',
  redirectTo = '/chat',
  className
}) => {
  const [mode, setMode] = useState<'login' | 'register'>(defaultMode);
  const router = useRouter();

  const handleSuccess = () => {
    router.push(redirectTo);
  };

  const switchToLogin = () => setMode('login');
  const switchToRegister = () => setMode('register');

  return (
    <div className={`min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50 flex items-center justify-center p-4 ${className}`}>
      {/* Home Button */}
      <div className="absolute top-6 left-6 z-10">
        <HomeButton 
          variant="secondary" 
          size="md"
          style="home"
          className="bg-white/80 backdrop-blur-sm border border-white/20 shadow-lg hover:shadow-xl hover:bg-white/90"
        />
      </div>
      
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        {/* Left side - Features */}
        <div className="hidden lg:block space-y-8">
          <div className="text-center lg:text-left">
            <h1 className="text-4xl lg:text-5xl font-bold text-secondary-900 mb-4">
              Welcome to ICD-11
              <span className="block text-primary-600">Mental Health Assistant</span>
            </h1>
            <p className="text-lg text-secondary-600 mb-8">
              Your trusted companion for mental health support and guidance
            </p>
          </div>

          <div className="space-y-6">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <Shield className="h-6 w-6 text-primary-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                  Professional Support
                </h3>
                <p className="text-secondary-600">
                  Get evidence-based guidance from our advanced AI system trained on ICD-11 standards
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-12 h-12 bg-mental-100 rounded-lg flex items-center justify-center">
                <Heart className="h-6 w-6 text-mental-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                  Empathetic Care
                </h3>
                <p className="text-secondary-600">
                  Experience compassionate and understanding responses tailored to your needs
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-12 h-12 bg-secondary-100 rounded-lg flex items-center justify-center">
                <Brain className="h-6 w-6 text-secondary-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                  Intelligent Insights
                </h3>
                <p className="text-secondary-600">
                  Receive personalized recommendations and follow-up suggestions
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white/50 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-success-100 rounded-full flex items-center justify-center">
                <Shield className="h-6 w-6 text-success-600" />
              </div>
              <div>
                <h4 className="font-semibold text-secondary-900">
                  Safe & Secure
                </h4>
                <p className="text-sm text-secondary-600">
                  Your privacy and data security are our top priorities
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Auth Form */}
        <div className="bg-white rounded-2xl shadow-xl p-8 lg:p-12">
          {mode === 'login' ? (
            <LoginForm
              onSwitchToRegister={switchToRegister}
              onSuccess={handleSuccess}
            />
          ) : (
            <RegisterForm
              onSwitchToLogin={switchToLogin}
              onSuccess={handleSuccess}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default AuthPage; 