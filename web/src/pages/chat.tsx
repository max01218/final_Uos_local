import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import ChatWindow from '../components/ChatWindow';
import ChatInput from '../components/ChatInput';
import FeedbackButton, { FeedbackData } from '../components/FeedbackButton';

interface Message {
  role: 'user' | 'assistant';
  content: string;
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
  const [lastQuestion, setLastQuestion] = useState('');
  const [lastAnswer, setLastAnswer] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const chatWindowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    // Get tone from URL parameter
    if (type && typeof type === 'string') {
      setCurrentTone(type);
    }
  }, [router.query]);

  async function handleSend(message: string) {
    if (!message.trim()) return;

    const userMessage: Message = { role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setLastQuestion(message);
    setShowFeedback(false);
    
    const filteredHistory = messages.slice(-4);
    
    try {
      // Direct call to backend API through SSH port forwarding
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
        const assistantMessage: Message = { 
          role: 'assistant', 
          content: data.answer 
        };
        setMessages(prev => [...prev, assistantMessage]);
        setLastAnswer(data.answer);
        setShowFeedback(true);
      }
    } catch (e) {
      console.error('Error sending message:', e);
      const errorMessage: Message = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please check if the SSH tunnel is active and the API server is running.' 
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

  const handleFeedback = async (feedbackData: FeedbackData) => {
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...feedbackData,
          question: lastQuestion,
          answer: lastAnswer,
          tone: currentTone
        }),
      });

      if (response.ok) {
        console.log('Feedback submitted successfully');
        setShowFeedback(false);
      } else {
        console.error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div style={{ height: '100vh', backgroundColor: '#ffffff' }}>
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        
        {/* Header */}
        <div style={{
          backgroundColor: '#f0f0f0',
          color: '#000000',
          padding: '15px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #cccccc'
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

        {/* Chat Area */}
        <div ref={chatWindowRef} style={{ 
          height: '450px', 
          overflowY: 'auto', 
          backgroundColor: '#ffffff', 
          padding: '20px', 
          marginBottom: '15px', 
          border: '1px solid #e0e0e0'
        }}>
          <ChatWindow messages={messages} />
          {loading && (
            <div style={{ color: '#666666', display: 'flex', alignItems: 'center', gap: 8, padding: '15px 0' }}>
              <div style={{ 
                width: 16, 
                height: 16, 
                border: '2px solid #e0e0e0', 
                borderTop: '2px solid #000000', 
                borderRadius: '50%', 
                animation: 'spin 1s linear infinite' 
              }}></div>
              <span>AI is responding...</span>
            </div>
          )}
        </div>
        
        {/* Input Area */}
        <div style={{ padding: '15px', borderTop: '1px solid #e0e0e0' }}>
          <ChatInput onSend={handleSend} disabled={loading} />
          
          {/* Feedback Area */}
          {showFeedback && (
            <div style={{ marginTop: '15px', padding: '15px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#333' }}>
                How was this response?
              </h4>
              <FeedbackButton onSubmit={handleFeedback} />
            </div>
          )}
        </div>
        
      </div>
      
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
} 
