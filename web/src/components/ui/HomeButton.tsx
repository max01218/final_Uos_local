import React from 'react';
import { useRouter } from 'next/router';
import { Home, ArrowLeft } from 'lucide-react';
import Button from './Button';

interface HomeButtonProps {
  className?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  style?: 'home' | 'back';
}

const HomeButton: React.FC<HomeButtonProps> = ({ 
  className = '', 
  variant = 'ghost',
  size = 'sm',
  showText = true,
  style = 'home'
}) => {
  const router = useRouter();

  const handleGoHome = () => {
    router.push('/');
  };

  const getButtonContent = () => {
    if (style === 'back') {
      return (
        <>
          <ArrowLeft className="h-4 w-4" />
          {showText && <span className="hidden sm:inline">Back</span>}
        </>
      );
    }
    
    return (
      <>
        <Home className="h-4 w-4" />
        {showText && <span className="hidden sm:inline">Home</span>}
      </>
    );
  };

  return (
    <Button
      onClick={handleGoHome}
      variant={variant}
      size={size}
      className={`flex items-center gap-2 transition-all duration-300 hover:scale-105 hover:shadow-lg active:scale-95 ${className}`}
    >
      {getButtonContent()}
    </Button>
  );
};

export default HomeButton; 