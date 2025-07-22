import React, { useState } from 'react';
import { useRouter } from 'next/router';

const TONE_OPTIONS = [
  { value: 'empathetic_professional', label: 'Empathetic + Professional' },
  { value: 'professional', label: 'Professional / Academic' },
  { value: 'caring', label: 'Caring / Empathetic' },
  { value: 'concise', label: 'Concise / Direct' },
  { value: 'casual', label: 'Casual / Conversational' },
];

export default function Home() {
  const router = useRouter();
  const [selectedTone, setSelectedTone] = useState('empathetic_professional');

  function handleToneChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setSelectedTone(e.target.value);
  }

  function startChat() {
    router.push(`/chat?type=${selectedTone}`);
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#ffffff',
      padding: '40px 20px'
    }}>
      <div style={{
        maxWidth: '400px',
        margin: '0 auto',
        textAlign: 'center'
      }}>
        <h1 style={{ 
          color: '#000000', 
          marginBottom: '40px'
        }}>
          Mental Health Assistant
        </h1>
        
        <div style={{ marginBottom: '30px' }}>
          <label style={{ 
            display: 'block',
            color: '#000000', 
            marginBottom: '10px',
            fontSize: '16px'
          }}>
            Response Style:
          </label>
          <select 
            value={selectedTone} 
            onChange={handleToneChange}
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '16px',
              border: '1px solid #cccccc',
              backgroundColor: '#ffffff',
              color: '#000000'
            }}
          >
            {TONE_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        
        <button 
          onClick={startChat}
          style={{
            padding: '15px 30px',
            backgroundColor: '#e0e0e0',
            color: '#000000',
            border: '1px solid #cccccc',
            cursor: 'pointer',
            width: '100%',
            fontSize: '16px',
            marginBottom: '15px'
          }}
        >
          Start Chat
        </button>
        
        <button 
          onClick={() => router.push('/feedback-dashboard')}
          style={{
            padding: '10px 20px',
            backgroundColor: '#f0f0f0',
            color: '#666666',
            border: '1px solid #cccccc',
            cursor: 'pointer',
            width: '100%',
            fontSize: '14px'
          }}
        >
          View Feedback Dashboard
        </button>
      </div>
    </div>
  );
} 