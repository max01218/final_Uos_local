import React, { useState } from 'react';

const TONE_OPTIONS = [
  { value: 'professional', label: 'Professional / Academic' },
  { value: 'caring', label: 'Caring / Empathetic' },
  { value: 'concise', label: 'Concise / Direct' },
  { value: 'casual', label: 'Casual / Conversational' },
];

export default function PromptPanel({ onChange }: { onChange?: (data: { type: string, info: string }) => void }) {
  const [type, setType] = useState('professional');
  const [info, setInfo] = useState('');

  function handleTypeChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setType(e.target.value);
    onChange?.({ type: e.target.value, info });
  }

  function handleInfoChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInfo(e.target.value);
    onChange?.({ type, info: e.target.value });
  }

  return (
    <div className="panel prompt-panel">
      <h3>Prompt</h3>
      <div className="prompt-section">
        <label>Type (Answer Tone)</label>
        <select value={type} onChange={handleTypeChange}>
          {TONE_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      <div className="prompt-section">
        <label>Basic information</label>
        <textarea
          placeholder="Enter basic info or symptoms..."
          rows={4}
          value={info}
          onChange={handleInfoChange}
        />
      </div>
    </div>
  );
} 