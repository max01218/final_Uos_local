import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { Heart, Shield, Users, Sparkles, ArrowRight, Leaf, Sun, Moon, History } from 'lucide-react';
import { useAuth } from '@/lib/AuthContext';
import UserMenu from '@/components/auth/UserMenu';
import Button from '@/components/ui/Button';
import HomeButton from '@/components/ui/HomeButton';

const TONE_OPTIONS = [
  { 
    value: 'empathetic_professional', 
    label: 'Balanced Professional', 
    description: 'Professional responses with empathy and understanding',
    icon: Heart,
    color: 'from-mental-500 to-mental-600'
  },
  { 
    value: 'caring', 
    label: 'Warm & Caring', 
    description: 'Conversations filled with warmth and care',
    icon: Shield,
    color: 'from-primary-500 to-primary-600'
  },
  { 
    value: 'professional', 
    label: 'Professional Guidance', 
    description: 'Structured and professional advice',
    icon: Users,
    color: 'from-secondary-600 to-secondary-700'
  },
];

const FEATURES = [
  {
    icon: Heart,
    title: 'Professional Mental Health Support',
    description: 'Expert diagnosis and advice based on ICD-11 standards'
  },
  {
    icon: Shield,
    title: 'Privacy & Security',
    description: 'Your privacy and safety are our top priorities'
  },
  {
    icon: Sparkles,
    title: 'Intelligent Personalization',
    description: 'Personalized responses tailored to your needs'
  },
  {
    icon: Leaf,
    title: 'Continuous Care',
    description: '24/7 mental health support available'
  }
];

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [selectedTone, setSelectedTone] = useState('empathetic_professional');
  const [hoveredTone, setHoveredTone] = useState<string | null>(null);

  function handleToneChange(value: string) {
    setSelectedTone(value);
  }

  function startChat() {
    if (isAuthenticated) {
      router.push(`/chat?type=${selectedTone}`);
    } else {
      router.push('/auth');
    }
  }

  const selectedOption = TONE_OPTIONS.find(option => option.value === selectedTone);

  return (
    <>
      <Head>
        <title>ICD-11 Mental Health Assistant - Professional Support</title>
        <meta name="description" content="Your professional mental health companion, providing warm, professional, and personalized psychological support based on ICD-11 standards" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-mental-50 via-white to-primary-50 relative overflow-hidden">
        {/* Background decorative elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-mental-200 to-mental-300 rounded-full opacity-20 animate-bounce-gentle"></div>
          <div className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-primary-200 to-primary-300 rounded-full opacity-15 animate-bounce-gentle" style={{animationDelay: '1s'}}></div>
          <div className="absolute bottom-20 left-1/4 w-20 h-20 bg-gradient-to-br from-secondary-200 to-secondary-300 rounded-full opacity-10 animate-bounce-gentle" style={{animationDelay: '2s'}}></div>
        </div>

        {/* Header with auth */}
        <header className="relative z-20 bg-white/80 backdrop-blur-sm border-b border-secondary-200">
          <div className="container-responsive">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-mental-500 to-mental-600 rounded-lg flex items-center justify-center">
                  <Heart className="h-4 w-4 text-white" />
                </div>
                <span className="text-lg font-semibold text-secondary-900">
                  Mental Health Assistant
                </span>
              </div>
              
              <div className="flex items-center gap-4">
                {isAuthenticated ? (
                  <UserMenu />
                ) : (
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      onClick={() => router.push('/auth')}
                    >
                      Sign In
                    </Button>
                    <Button
                      variant="primary"
                      onClick={() => router.push('/auth')}
                    >
                      Get Started
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        <div className="relative z-10 container-responsive py-12">
          {/* Header section */}
          <div className="text-center mb-16 animate-fade-in">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-mental-500 to-mental-600 rounded-full mb-6 shadow-soft">
              <Heart className="w-10 h-10 text-white" />
            </div>
            
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 gradient-text">
              Mental Health Assistant
            </h1>
            
            <p className="text-xl md:text-2xl text-secondary-600 mb-8 max-w-3xl mx-auto leading-relaxed">
              Your professional mental health companion, providing warm, professional, and personalized psychological support
            </p>
            
            <div className="flex items-center justify-center gap-4 text-secondary-500">
              <Sun className="w-5 h-5" />
              <span className="text-sm">24/7 Support Available</span>
              <Moon className="w-5 h-5" />
            </div>
          </div>

          {/* Main features section */}
          <div className="max-w-4xl mx-auto mb-16">
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              {FEATURES.map((feature, index) => (
                <div 
                  key={index}
                  className="card text-center group hover:shadow-medium transition-all duration-300 hover:-translate-y-1"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-mental-100 to-mental-200 rounded-xl mb-4 group-hover:scale-110 transition-transform">
                    <feature.icon className="w-6 h-6 text-mental-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-secondary-600 text-sm leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>

            {/* Conversation style selection */}
            <div className="card mb-8">
              <h2 className="text-2xl font-semibold text-secondary-900 mb-6 text-center">
                Choose Your Conversation Style
              </h2>
              
              <div className="grid md:grid-cols-3 gap-4">
                {TONE_OPTIONS.map((option) => {
                  const IconComponent = option.icon;
                  const isSelected = selectedTone === option.value;
                  const isHovered = hoveredTone === option.value;
                  
                  return (
                    <button
                      key={option.value}
                      onClick={() => handleToneChange(option.value)}
                      onMouseEnter={() => setHoveredTone(option.value)}
                      onMouseLeave={() => setHoveredTone(null)}
                      className={`
                        relative p-6 rounded-xl border-2 transition-all duration-300 text-left group
                        ${isSelected 
                          ? 'border-mental-500 bg-gradient-to-br from-mental-50 to-mental-100 shadow-soft' 
                          : 'border-secondary-200 bg-white hover:border-mental-300 hover:shadow-soft'
                        }
                      `}
                    >
                      <div className={`
                        inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4 transition-all duration-300
                        ${isSelected 
                          ? `bg-gradient-to-br ${option.color}` 
                          : 'bg-secondary-100 group-hover:bg-mental-100'
                        }
                      `}>
                        <IconComponent className={`w-6 h-6 transition-colors duration-300 ${
                          isSelected ? 'text-white' : 'text-secondary-600 group-hover:text-mental-600'
                        }`} />
                      </div>
                      
                      <h3 className={`font-semibold mb-2 transition-colors duration-300 ${
                        isSelected ? 'text-mental-700' : 'text-secondary-900'
                      }`}>
                        {option.label}
                      </h3>
                      
                      <p className="text-sm text-secondary-600 leading-relaxed">
                        {option.description}
                      </p>
                      
                      {isSelected && (
                        <div className="absolute top-3 right-3 w-6 h-6 bg-gradient-to-br from-mental-500 to-mental-600 rounded-full flex items-center justify-center">
                          <div className="w-2 h-2 bg-white rounded-full"></div>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Start conversation button */}
            <div className="text-center">
              <button 
                onClick={startChat}
                className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-mental-500 to-mental-600 hover:from-mental-600 hover:to-mental-700 text-white font-semibold rounded-xl shadow-soft hover:shadow-medium transition-all duration-300 hover:-translate-y-1 text-lg"
              >
                {isAuthenticated ? 'Start Conversation' : 'Get Started'}
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              
              {isAuthenticated && (
                <div className="mt-4">
                  <Button
                    variant="secondary"
                    onClick={() => router.push('/conversation-history')}
                    className="inline-flex items-center gap-2"
                  >
                    <History className="h-4 w-4" />
                    View Conversation History
                  </Button>
                </div>
              )}
              
              <p className="text-secondary-500 text-sm mt-4">
                {isAuthenticated 
                  ? 'Choose the conversation style that suits you and begin your mental health journey'
                  : 'Sign up to start your personalized mental health journey'
                }
              </p>
            </div>
          </div>

          {/* Footer information */}
          <div className="text-center text-secondary-500 text-sm">
            <p className="mb-2">
              Your mental health is our primary concern
            </p>
            <p>
              For emergency assistance, please call a mental health hotline or contact professional medical services
            </p>
          </div>
        </div>
      </div>
    </>
  );
} 