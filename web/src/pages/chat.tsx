import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

const TONE_LABELS = {
  'professional': 'Professional / Academic',
  'caring': 'Caring / Empathetic',
  'concise': 'Concise / Direct',
  'casual': 'Casual / Conversational',
  'empathetic_professional': 'Empathetic + Professional'
};

export default function ChatPage() {
  const router = useRouter();
  const { type } = router.query;
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentTone, setCurrentTone] = useState('empathetic_professional');
  const chatWindowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (type && typeof type === 'string') {
      setCurrentTone(type);
    }
  }, [router.query]);

  const generateMessageId = () => {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  async function handleSend(message: string) {
    if (!message.trim()) return;

    const userMessageId = generateMessageId();
    const userMessage: Message = { 
      id: userMessageId,
      role: 'user', 
      content: message,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    
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
          history: filteredHistory
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      } else {
        const data = await response.json();
        const assistantMessageId = generateMessageId();
        const assistantMessage: Message = { 
          id: assistantMessageId,
          role: 'assistant', 
          content: data.answer,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (e) {
      console.error('Error sending message:', e);
      const errorMessageId = generateMessageId();
      const errorMessage: Message = { 
        id: errorMessageId,
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please check if the SSH tunnel is active and the API server is running.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }

  const goBack = () => {
    router.push('/');
  };

  const changeTone = () => {
    router.push('/');
  };

  return (
    <div style={{ height: '100vh', backgroundColor: '#ffffff', display: 'flex', flexDirection: 'column' }}>
      
      {/* Header */}
      <div style={{
        backgroundColor: '#f0f0f0',
        color: '#000000',
        padding: '15px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #cccccc',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <button 
            onClick={goBack}
            style={{
              backgroundColor: '#e0e0e0',
              color: '#000000',
              border: '1px solid #cccccc',
              padding: '8px 12px',
              cursor: 'pointer',
              marginRight: '15px'
            }}
          >
            Back
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: '20px' }}>Mental Health Assistant</h1>
            <p style={{ 
              margin: '5px 0 0 0', 
              fontSize: '14px', 
              color: '#666666',
              display: 'flex',
              alignItems: 'center'
            }}>
              <span style={{ marginRight: '8px' }}>Current Style:</span>
              <span style={{ 
                backgroundColor: '#e0e0e0', 
                padding: '2px 8px', 
                border: '1px solid #cccccc',
                fontSize: '12px'
              }}>
                {TONE_LABELS[currentTone as keyof typeof TONE_LABELS] || currentTone}
              </span>
            </p>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            onClick={changeTone}
            style={{
              backgroundColor: '#e0e0e0',
              color: '#000000',
              border: '1px solid #cccccc',
              padding: '8px 12px',
              cursor: 'pointer'
            }}
          >
            Change Style
          </button>
        </div>
      </div>

      {/* Chat Area with Input */}
      <div style={{ 
        flex: 1,
        position: 'relative',
        backgroundColor: '#ffffff'
      }}>
        <div ref={chatWindowRef} style={{ 
          height: '100%',
          overflowY: 'auto', 
          padding: '20px',
          paddingBottom: '100px'
        }}>
          {messages.map((message) => (
            <div
              key={message.id}
              style={{
                marginBottom: '15px',
                display: 'flex',
                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start'
              }}
            >
              <div
                style={{
                  maxWidth: '70%',
                  padding: '12px 16px',
                  borderRadius: '18px',
                  backgroundColor: message.role === 'user' ? '#007bff' : '#f1f1f1',
                  color: message.role === 'user' ? '#ffffff' : '#000000',
                  wordWrap: 'break-word'
                }}
              >
                {message.content}
              </div>
            </div>
          ))}
          
          {loading && (
            <div style={{
              display: 'flex',
              justifyContent: 'flex-start',
              marginBottom: '15px'
            }}>
              <div style={{
                padding: '12px 16px',
                borderRadius: '18px',
                backgroundColor: '#f1f1f1',
                color: '#666666'
              }}>
                Thinking...
              </div>
            </div>
          )}
        </div>

        {/* Input Area - positioned at bottom */}
        <div style={{ 
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          right: '20px',
          backgroundColor: '#ffffff',
          padding: '15px',
          border: '1px solid #e0e0e0',
          borderRadius: '10px',
          display: 'flex',
          gap: '10px',
          zIndex: 1000,
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
          <input
            type="text"
            placeholder="Type your message..."
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const target = e.target as HTMLInputElement;
                if (target.value.trim()) {
                  handleSend(target.value);
                  target.value = '';
                }
              }
            }}
            disabled={loading}
            style={{
              flex: 1,
              padding: '12px 16px',
              border: '1px solid #cccccc',
              borderRadius: '20px',
              fontSize: '14px',
              outline: 'none'
            }}
          />
          <button
            onClick={(e) => {
              const input = e.currentTarget.previousElementSibling as HTMLInputElement;
              if (input.value.trim()) {
                handleSend(input.value);
                input.value = '';
              }
            }}
            disabled={loading}
            style={{
              padding: '12px 20px',
              backgroundColor: '#007bff',
              color: '#ffffff',
              border: 'none',
              borderRadius: '20px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '14px'
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
} 
