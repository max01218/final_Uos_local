import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

import { Toaster, toast } from 'react-hot-toast';
import { Message, ToneType, ToneConfig } from '@/types';
import { generateId } from '@/lib/utils';
import ChatWindow from '@/components/chat/ChatWindow';
import ChatInput from '@/components/chat/ChatInput';
import Button from '@/components/ui/Button';
import HomeButton from '@/components/ui/HomeButton';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import UserMenu from '@/components/auth/UserMenu';
import { 
  ArrowLeft, 
  Settings, 
  Heart, 
  Shield, 
  Brain,
  Menu,
  X
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Tone configurations
const TONE_CONFIGS: Record<ToneType, ToneConfig> = {
  professional: {
    id: 'professional',
    label: 'Professional',
    description: 'Provide professional, objective advice',
    icon: '👨‍⚕️',
    color: 'bg-blue-500'
  },
  caring: {
    id: 'caring',
    label: 'Caring',
    description: 'Warm, supportive communication style',
    icon: '💝',
    color: 'bg-pink-500'
  },
  empathetic_professional: {
    id: 'empathetic_professional',
    label: 'Balanced',
    description: 'Perfect combination of professional and caring',
    icon: '🤝',
    color: 'bg-purple-500'
  }
};

function ChatPageContent() {
  const router = useRouter();
  const { type } = router.query;
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentTone, setCurrentTone] = useState<ToneType>('empathetic_professional');
  const [showSettings, setShowSettings] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Set tone from URL parameters
  useEffect(() => {
    if (type && typeof type === 'string' && type in TONE_CONFIGS) {
      setCurrentTone(type as ToneType);
    }
  }, [router.query]);

  // Handle sending messages
  const handleSend = useCallback(async (message: string, metadata?: any) => {
    if (!message.trim()) return;

    const userMessageId = generateId();
    const userMessage: Message = {
      id: userMessageId,
      role: 'user',
      content: message,
      timestamp: new Date(),
      metadata: metadata
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setError(null);

    // Limit history messages for better performance
    const filteredHistory = messages.slice(-4);
    
    try {
      const response = await fetch('/api/empathetic_professional', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          type: currentTone,
          tone: currentTone,
          history: filteredHistory,
          metadata: {
            userAgent: navigator.userAgent,
            sessionId: generateId(),
            timestamp: Date.now()
          }
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const assistantMessageId = generateId();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        metadata: {
          confidence: data.confidence,
          processing_time: data.processing_time,
          safety_alerts: data.safety_alerts,
          emotion_analysis: data.emotion_analysis,
          follow_up_suggestions: data.follow_up_suggestions
        }
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      toast.success('Message sent');
      
    } catch (e) {
      console.error('Error sending message:', e);
      const errorMessage = e instanceof Error ? e.message : 'Unknown error';
      setError(errorMessage);
      
      const errorMessageId = generateId();
      const errorMsg: Message = {
        id: errorMessageId,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please check if the SSH tunnel is active and the API server is running.',
        timestamp: new Date(),
        type: 'error'
      };
      setMessages(prev => [...prev, errorMsg]);
      toast.error('Send failed, please try again');
    } finally {
      setLoading(false);
    }
  }, [messages, currentTone]);

  // Handle feedback
  const handleFeedback = useCallback((messageId: string, type: 'positive' | 'negative') => {
    // Here you can send feedback to the backend
    console.log('Feedback:', { messageId, type });
    toast.success(type === 'positive' ? 'Thank you for your feedback!' : 'We will continue to improve');
  }, []);

  // Handle import
  const handleImport = useCallback((importedMessages: Message[]) => {
    setMessages(importedMessages);
    toast.success('Chat history imported');
  }, []);

  // Go back to home
  const goBack = () => {
    router.push('/');
  };

  // Change tone
  const changeTone = () => {
    router.push('/tone-select');
  };

  // Clear conversation
  const clearConversation = () => {
    if (confirm('Are you sure you want to clear all conversations? This action cannot be undone.')) {
      setMessages([]);
      toast.success('Conversation cleared');
    }
  };

  const currentToneConfig = TONE_CONFIGS[currentTone];

  return (
    <div className="min-h-screen bg-secondary-50">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      
      {/* Header */}
      <header className="bg-white border-b border-secondary-200 shadow-soft animate-slide-down">
        <div className="container-responsive">
          <div className="flex items-center justify-between h-16">
            {/* Left side */}
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={goBack}
                className="md:hidden"
                aria-label="Back"
              >
                <ArrowLeft className="h-5 w-5" />
              </Button>
              
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-mental-500 rounded-lg flex items-center justify-center">
                  <Heart className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-secondary-900">
                    Mental Health Assistant
                  </h1>
                  <div className="flex items-center gap-2 text-sm text-secondary-600">
                    <span>Style:</span>
                    <span className={cn(
                      'px-2 py-1 rounded-full text-xs font-medium',
                      currentToneConfig.color,
                      'text-white'
                    )}>
                      {currentToneConfig.icon} {currentToneConfig.label}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right side */}
            <div className="flex items-center gap-2">
              <HomeButton 
                variant="ghost" 
                size="sm"
                style="home"
                className="hover:bg-secondary-100 hover:text-primary-600"
              />
              
              <Button
                variant="ghost"
                onClick={changeTone}
                className="hidden md:flex"
              >
                Change Style
              </Button>
              
              <Button
                variant="ghost"
                onClick={() => setShowSettings(!showSettings)}
                aria-label="Settings"
              >
                <Settings className="h-5 w-5" />
              </Button>
              
              <UserMenu />
              
              <Button
                variant="ghost"
                onClick={() => setShowMobileMenu(!showMobileMenu)}
                className="md:hidden"
                aria-label="Menu"
              >
                {showMobileMenu ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile menu */}
      {showMobileMenu && (
        <div className="md:hidden bg-white border-b border-secondary-200 animate-slide-down">
          <div className="container-responsive py-4 space-y-2">
            <Button
              variant="ghost"
              onClick={changeTone}
              className="w-full justify-start"
            >
              Change Style
            </Button>
            <Button
              variant="ghost"
              onClick={clearConversation}
              className="w-full justify-start text-error-600"
            >
              Clear Conversation
            </Button>
          </div>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <div className="bg-white border-b border-secondary-200 animate-slide-down">
          <div className="container-responsive py-4">
            <h3 className="text-sm font-medium text-secondary-900 mb-3">Settings</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="h-4 w-4 text-success-600" />
                  <span className="text-sm font-medium">Safety Alerts</span>
                </div>
                <p className="text-xs text-secondary-600">
                  Automatically detect potential safety risks and provide help
                </p>
              </div>
              
              <div className="p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="h-4 w-4 text-mental-600" />
                  <span className="text-sm font-medium">Emotion Detection</span>
                </div>
                <p className="text-xs text-secondary-600">
                  Analyze user emotions and provide personalized suggestions
                </p>
              </div>
              
              <div className="p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Heart className="h-4 w-4 text-primary-600" />
                  <span className="text-sm font-medium">Caring Mode</span>
                </div>
                <p className="text-xs text-secondary-600">
                  Provide warm, supportive communication experience
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="container-responsive py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Chat area */}
          <div className="lg:col-span-3">
            <ChatWindow
              messages={messages}
              isLoading={loading}
              onFeedback={handleFeedback}
              onImport={handleImport}
              className="h-[calc(100vh-200px)]"
            />
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            {/* Quick Actions */}
            <div className="card">
              <h3 className="text-sm font-medium text-secondary-900 mb-3">Quick Actions</h3>
              <div className="space-y-2">
                <Button
                  variant="secondary"
                  onClick={clearConversation}
                  className="w-full justify-start"
                >
                  Clear Conversation
                </Button>
              </div>
            </div>

            {/* Current Style Info */}
            <div className="card">
              <h3 className="text-sm font-medium text-secondary-900 mb-3">Current Style</h3>
              <div className="flex items-center gap-3 p-3 bg-secondary-50 rounded-lg">
                <div className={cn(
                  'w-10 h-10 rounded-full flex items-center justify-center text-lg',
                  currentToneConfig.color
                )}>
                  {currentToneConfig.icon}
                </div>
                <div>
                  <div className="font-medium text-secondary-900">
                    {currentToneConfig.label}
                  </div>
                  <div className="text-xs text-secondary-600">
                    {currentToneConfig.description}
                  </div>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="card">
              <h3 className="text-sm font-medium text-secondary-900 mb-3">Conversation Stats</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-secondary-600">Total Messages</span>
                  <span className="font-medium">{messages.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600">User Messages</span>
                  <span className="font-medium">
                    {messages.filter(m => m.role === 'user').length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600">Assistant Replies</span>
                  <span className="font-medium">
                    {messages.filter(m => m.role === 'assistant').length}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Input area */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-secondary-200 p-4">
        <div className="container-responsive">
          <ChatInput
            onSend={handleSend}
            disabled={loading}
            placeholder="Share your thoughts..."
            showEmotionDetection={true}
            showSafetyWarnings={true}
          />
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="fixed top-4 right-4 bg-error-50 border border-error-200 rounded-lg p-4 max-w-sm animate-slide-in">
          <div className="flex items-start gap-3">
            <div className="w-5 h-5 bg-error-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-error-600 text-xs">!</span>
            </div>
            <div>
              <h4 className="text-sm font-medium text-error-800 mb-1">Connection Error</h4>
              <p className="text-xs text-error-700">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  return (
    <>
      <Head>
        <title>Chat - ICD-11 Mental Health Assistant</title>
        <meta name="description" content="Chat with your personalized mental health assistant" />
      </Head>
      
      <ProtectedRoute>
        <ChatPageContent />
      </ProtectedRoute>
    </>
  );
} 
