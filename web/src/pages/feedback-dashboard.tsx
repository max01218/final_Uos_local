import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

interface FeedbackData {
  satisfaction: number;
  empathy: number;
  accuracy: number;
  safety: number;
  comment: string;
  question?: string;
  answer?: string;
  tone?: string;
  timestamp: string;
}

export default function FeedbackDashboard() {
  const router = useRouter();
  const [feedbacks, setFeedbacks] = useState<FeedbackData[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    avgSatisfaction: 0,
    avgEmpathy: 0,
    avgAccuracy: 0,
    avgSafety: 0
  });

  useEffect(() => {
    loadFeedbacks();
  }, []);

  const loadFeedbacks = async () => {
    try {
      const response = await fetch('/api/feedback');
      if (response.ok) {
        const data = await response.json();
        setFeedbacks(data.feedbacks || []);
        calculateStats(data.feedbacks || []);
      }
    } catch (error) {
      console.error('Error loading feedbacks:', error);
    } finally {
      setLoading(false);
    }
  };

  const calculateStats = (data: FeedbackData[]) => {
    if (data.length === 0) return;

    const total = data.length;
    const avgSatisfaction = data.reduce((sum, f) => sum + f.satisfaction, 0) / total;
    const avgEmpathy = data.reduce((sum, f) => sum + f.empathy, 0) / total;
    const avgAccuracy = data.reduce((sum, f) => sum + f.accuracy, 0) / total;
    const avgSafety = data.reduce((sum, f) => sum + f.safety, 0) / total;

    setStats({
      total,
      avgSatisfaction: Math.round(avgSatisfaction * 100) / 100,
      avgEmpathy: Math.round(avgEmpathy * 100) / 100,
      avgAccuracy: Math.round(avgAccuracy * 100) / 100,
      avgSafety: Math.round(avgSafety * 100) / 100
    });
  };

  const goBack = () => {
    router.push('/');
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        Loading feedback data...
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
        <button onClick={goBack} style={{ marginRight: '15px', padding: '8px 12px' }}>
          Back
        </button>
        <h1>Feedback Dashboard</h1>
      </div>

      {/* Statistics */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '15px', 
        marginBottom: '30px' 
      }}>
        <div style={{ padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
          <h3>Total Feedbacks</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{stats.total}</p>
        </div>
        <div style={{ padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
          <h3>Avg Satisfaction</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{stats.avgSatisfaction}/5</p>
        </div>
        <div style={{ padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
          <h3>Avg Empathy</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{stats.avgEmpathy}/5</p>
        </div>
        <div style={{ padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
          <h3>Avg Accuracy</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{stats.avgAccuracy}/5</p>
        </div>
        <div style={{ padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
          <h3>Avg Safety</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{stats.avgSafety}/5</p>
        </div>
      </div>

      {/* Feedback List */}
      <div>
        <h2>Recent Feedbacks</h2>
        {feedbacks.length === 0 ? (
          <p>No feedbacks yet.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {feedbacks.slice().reverse().map((feedback, index) => (
              <div key={index} style={{ 
                border: '1px solid #ddd', 
                borderRadius: '8px', 
                padding: '15px',
                backgroundColor: '#fff'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                  <span style={{ color: '#666', fontSize: '14px' }}>
                    {new Date(feedback.timestamp).toLocaleString()}
                  </span>
                  <span style={{ 
                    backgroundColor: '#e0e0e0', 
                    padding: '2px 8px', 
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}>
                    {feedback.tone || 'empathetic_professional'}
                  </span>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '10px' }}>
                  <div>Satisfaction: <strong>{feedback.satisfaction}/5</strong></div>
                  <div>Empathy: <strong>{feedback.empathy}/5</strong></div>
                  <div>Accuracy: <strong>{feedback.accuracy}/5</strong></div>
                  <div>Safety: <strong>{feedback.safety}/5</strong></div>
                </div>
                
                {feedback.question && (
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Question:</strong> {feedback.question}
                  </div>
                )}
                
                {feedback.answer && (
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Answer:</strong> {feedback.answer.substring(0, 200)}...
                  </div>
                )}
                
                {feedback.comment && (
                  <div>
                    <strong>Comment:</strong> {feedback.comment}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 