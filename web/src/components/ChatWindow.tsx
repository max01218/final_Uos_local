import React from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatWindow({ messages }: { messages: Message[] }) {
  return (
    <div>
      {messages.map((msg, idx) => (
        <div key={idx} style={{ marginBottom: 12, textAlign: msg.role === 'user' ? 'right' : 'left' }}>
          <span style={{
            display: 'inline-block',
            backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#f5f5f5',
            color: '#000000',
            border: msg.role === 'user' ? '1px solid #2196f3' : '1px solid #cccccc',
            padding: '10px 16px',
            maxWidth: '70%',
            wordBreak: 'break-word',
            borderRadius: '12px',
            fontSize: '14px',
            lineHeight: '1.4',
          }}>
            {msg.content}
          </span>
        </div>
      ))}
    </div>
  );
} 