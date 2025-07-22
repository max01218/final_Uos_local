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
            backgroundColor: msg.role === 'user' ? '#f0f0f0' : '#f9f9f9',
            color: '#000000',
            border: '1px solid #cccccc',
            padding: '8px 14px',
            maxWidth: '70%',
            wordBreak: 'break-word',
          }}>
            {msg.content}
          </span>
        </div>
      ))}
    </div>
  );
} 