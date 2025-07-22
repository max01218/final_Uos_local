import React, { useState } from 'react';

export default function ChatInput({ onSend, disabled }: { onSend: (msg: string) => void, disabled?: boolean }) {
  const [value, setValue] = useState('');

  function handleSend() {
    if (value.trim()) {
      onSend(value.trim());
      setValue('');
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <textarea
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={2}
        style={{ 
          flex: 1, 
          resize: 'none', 
          border: '1px solid #cccccc', 
          padding: '8px',
          backgroundColor: '#ffffff'
        }}
        placeholder="Type your message..."
        disabled={disabled}
      />
      <button
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        style={{ 
          padding: '8px 18px', 
          backgroundColor: disabled || !value.trim() ? '#f0f0f0' : '#e0e0e0', 
          color: '#000000', 
          border: '1px solid #cccccc',
          cursor: disabled || !value.trim() ? 'not-allowed' : 'pointer'
        }}
      >
        Send
      </button>
    </div>
  );
} 