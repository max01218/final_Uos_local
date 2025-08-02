import React from 'react';
import { useRouter } from 'next/router';
import HomeButton from '@/components/ui/HomeButton';

const TONE_OPTIONS = [
  {
    id: 'empathetic_professional',
    name: 'Balanced',
    description: 'Caring and professional'
  },
  {
    id: 'caring',
    name: 'Caring',
    description: 'Warm and supportive'
  },
  {
    id: 'professional',
    name: 'Professional',
    description: 'Clinical and direct'
  }
];

export default function ToneSelectPage() {
  const router = useRouter();

  const selectTone = (toneId: string) => {
    router.push(`/chat?type=${toneId}`);
  };

  const goBack = () => {
    router.push('/');
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f5f5f5',
      padding: '20px'
    }}>
      <div style={{
        maxWidth: '600px',
        margin: '0 auto',
        backgroundColor: '#ffffff',
        borderRadius: '10px',
        padding: '30px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
      }}>
                 <div style={{ marginBottom: '30px' }}>
           <div style={{ marginBottom: '20px' }}>
             <HomeButton 
               variant="secondary" 
               size="md"
               style="back"
               className="shadow-md"
             />
           </div>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '24px' }}>
            Choose Style
          </h1>
          <p style={{ margin: 0, color: '#666666' }}>
            Select how you want the assistant to respond
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          {TONE_OPTIONS.map((tone) => (
            <button
              key={tone.id}
              onClick={() => selectTone(tone.id)}
              style={{
                backgroundColor: '#ffffff',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
                padding: '20px',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column',
                gap: '5px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#007bff';
                e.currentTarget.style.backgroundColor = '#f8f9fa';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#e0e0e0';
                e.currentTarget.style.backgroundColor = '#ffffff';
              }}
            >
              <div style={{ 
                fontSize: '18px', 
                fontWeight: 'bold',
                color: '#000000'
              }}>
                {tone.name}
              </div>
              <div style={{ 
                fontSize: '14px', 
                color: '#666666'
              }}>
                {tone.description}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
} 