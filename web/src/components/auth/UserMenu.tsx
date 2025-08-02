import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '@/lib/AuthContext';
import Button from '@/components/ui/Button';
import { User, Settings, LogOut, ChevronDown, Shield } from 'lucide-react';

interface UserMenuProps {
  className?: string;
}

const UserMenu: React.FC<UserMenuProps> = ({ className }) => {
  const { user, logout, isAuthenticated } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
      setIsOpen(false);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  if (!isAuthenticated || !user) {
    return null;
  }

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className={`relative ${className}`} ref={menuRef}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-secondary-100 transition-colors"
      >
        <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
          <span className="text-sm font-medium text-primary-700">
            {getInitials(user.name)}
          </span>
        </div>
        <div className="hidden sm:block text-left">
          <div className="text-sm font-medium text-secondary-900">
            {user.name}
          </div>
          <div className="text-xs text-secondary-500">
            {user.email}
          </div>
        </div>
        <ChevronDown className={`h-4 w-4 text-secondary-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-secondary-200 py-2 z-50">
          {/* User Info */}
          <div className="px-4 py-3 border-b border-secondary-100">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-primary-700">
                  {getInitials(user.name)}
                </span>
              </div>
              <div>
                <div className="text-sm font-medium text-secondary-900">
                  {user.name}
                </div>
                <div className="text-xs text-secondary-500">
                  {user.email}
                </div>
                {user.isVerified && (
                  <div className="flex items-center mt-1">
                    <Shield className="h-3 w-3 text-success-500 mr-1" />
                    <span className="text-xs text-success-600">Verified</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Menu Items */}
          <div className="py-1">
            <button
              onClick={() => {
                setIsOpen(false);
                // Navigate to profile/settings
              }}
              className="flex items-center w-full px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50 transition-colors"
            >
              <User className="h-4 w-4 mr-3" />
              Profile
            </button>
            
            <button
              onClick={() => {
                setIsOpen(false);
                // Navigate to settings
              }}
              className="flex items-center w-full px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50 transition-colors"
            >
              <Settings className="h-4 w-4 mr-3" />
              Settings
            </button>
          </div>

          {/* Divider */}
          <div className="border-t border-secondary-100 my-1" />

          {/* Logout */}
          <div className="py-1">
            <button
              onClick={handleLogout}
              className="flex items-center w-full px-4 py-2 text-sm text-error-600 hover:bg-error-50 transition-colors"
            >
              <LogOut className="h-4 w-4 mr-3" />
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserMenu; 