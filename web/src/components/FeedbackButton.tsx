import React, { useState } from 'react';

interface FeedbackButtonProps {
  onSubmit: (feedback: FeedbackData) => void;
}

export interface FeedbackData {
  satisfaction: number;
  empathy: number;
  accuracy: number;
  safety: number;
  comment: string;
}

const FeedbackButton: React.FC<FeedbackButtonProps> = ({ onSubmit }) => {
  const [showForm, setShowForm] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackData>({
    satisfaction: 0,
    empathy: 0,
    accuracy: 0,
    safety: 0,
    comment: '',
  });

  const handleChange = (field: keyof FeedbackData, value: string | number) => {
    setFeedback({ ...feedback, [field]: value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(feedback);
    setShowForm(false);
    setFeedback({ satisfaction: 0, empathy: 0, accuracy: 0, safety: 0, comment: '' });
  };

  return (
    <div>
      {!showForm ? (
        <button onClick={() => setShowForm(true)} style={{ marginTop: 12 }}>
          Give Feedback
        </button>
      ) : (
        <form onSubmit={handleSubmit} style={{ marginTop: 12, border: '1px solid #ccc', padding: 12, borderRadius: 8 }}>
          <div>
            <label>Satisfaction: </label>
            <input type="number" min={1} max={5} value={feedback.satisfaction} onChange={e => handleChange('satisfaction', Number(e.target.value))} required />
          </div>
          <div>
            <label>Empathy: </label>
            <input type="number" min={1} max={5} value={feedback.empathy} onChange={e => handleChange('empathy', Number(e.target.value))} required />
          </div>
          <div>
            <label>Accuracy: </label>
            <input type="number" min={1} max={5} value={feedback.accuracy} onChange={e => handleChange('accuracy', Number(e.target.value))} required />
          </div>
          <div>
            <label>Safety: </label>
            <input type="number" min={1} max={5} value={feedback.safety} onChange={e => handleChange('safety', Number(e.target.value))} required />
          </div>
          <div>
            <label>Comment: </label>
            <input type="text" value={feedback.comment} onChange={e => handleChange('comment', e.target.value)} />
          </div>
          <button type="submit" style={{ marginTop: 8 }}>Submit</button>
          <button type="button" onClick={() => setShowForm(false)} style={{ marginLeft: 8 }}>Cancel</button>
        </form>
      )}
    </div>
  );
};

export default FeedbackButton; 