import React from 'react';

const TONE_OPTIONS = [
  { value: 'professional', label: 'Professional / Academic' },
  { value: 'caring', label: 'Caring / Empathetic' },
  { value: 'concise', label: 'Concise / Direct' },
  { value: 'casual', label: 'Casual / Conversational' },
];

export default function ToneSelector({ onSelect }: { onSelect: (type: string) => void }) {
  const [selected, setSelected] = React.useState('professional');

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setSelected(e.target.value);
  }

  function handleStart() {
    onSelect(selected);
  }

  return (
    <div style={{ maxWidth: 400, margin: '20px auto', textAlign: 'center' }}>
      <h2>Select Answer Tone</h2>
      <select value={selected} onChange={handleChange} style={{ 
        fontSize: '16px', 
        padding: '8px', 
        width: '80%',
        border: '1px solid #cccccc',
        backgroundColor: '#ffffff'
      }}>
        {TONE_OPTIONS.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      <br /><br />
      <button onClick={handleStart} style={{ 
        padding: '12px 24px', 
        fontSize: '16px', 
        backgroundColor: '#e0e0e0', 
        color: '#000000', 
        border: '1px solid #cccccc', 
        cursor: 'pointer' 
      }}>
        Start Chat
      </button>
    </div>
  );
} 