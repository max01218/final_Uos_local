import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';

import { ToneType, ToneConfig } from '@/types';
import Button from '@/components/ui/Button';
import HomeButton from '@/components/ui/HomeButton';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import { cn } from '@/lib/utils';

const TONE_CONFIGS: Record<ToneType, ToneConfig> = {
  professional: {
    id: 'professional',
    label: 'Professional',
    description: 'Provide professional, objective advice',
    icon: '👨‍⚕️',
    color: 'bg-blue-500',
  },
  caring: {
    id: 'caring',
    label: 'Caring',
    description: 'Warm, supportive communication style',
    icon: '💝',
    color: 'bg-pink-500',
  },
  empathetic_professional: {
    id: 'empathetic_professional',
    label: 'Balanced',
    description: 'Balanced empathy with professional guidance',
    icon: '🤝',
    color: 'bg-purple-500',
  },
};

function ToneSelectPageContent() {
  const router = useRouter();

  const handleSelect = (tone: ToneType) => {
    router.push(`/chat?type=${encodeURIComponent(tone)}`);
  };

  const tones: ToneType[] = ['professional', 'caring', 'empathetic_professional'];

  return (
    <div className="min-h-screen bg-secondary-50">
      <Head>
        <title>Select Style - ICD-11 Mental Health Assistant</title>
        <meta name="description" content="Select conversation style" />
      </Head>

      <header className="bg-white border-b border-secondary-200 shadow-soft">
        <div className="container-responsive">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-mental-500 rounded-lg" />
              <h1 className="text-lg font-semibold text-secondary-900">Select Style</h1>
            </div>
            <div className="flex items-center gap-2">
              <HomeButton variant="ghost" size="sm" style="home" className="hover:bg-secondary-100 hover:text-primary-600" />
              <Button variant="ghost" onClick={() => router.push('/chat')}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container-responsive py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {tones.map((tone) => {
            const cfg = TONE_CONFIGS[tone];
            return (
              <div
                key={cfg.id}
                role="button"
                tabIndex={0}
                onClick={() => handleSelect(cfg.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleSelect(cfg.id);
                  }
                }}
                className={cn(
                  'group p-5 rounded-xl border border-secondary-200 bg-white hover:shadow-md transition-shadow text-left cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary-500',
                )}
                aria-label={`Select ${cfg.label} style`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={cn('w-10 h-10 rounded-full flex items-center justify-center text-lg', cfg.color)}>
                    {cfg.icon}
                  </div>
                  <div className="text-secondary-900 font-semibold">{cfg.label}</div>
                </div>
                <div className="text-sm text-secondary-600">{cfg.description}</div>
                <div className="mt-4">
                  <Button variant="primary" className="w-full">Use this style</Button>
                </div>
              </div>
            );
          })}
        </div>
      </main>
    </div>
  );
}

export default function ToneSelectPage() {
  return (
    <ProtectedRoute>
      <ToneSelectPageContent />
    </ProtectedRoute>
  );
}