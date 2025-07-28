import React, { useState, useEffect } from 'react';

interface FeedbackData {
  messageId: string;
  helpful: boolean | null;
  accuracy: number;
  empathy: number;
  safety: number;
  clarity: number;
  emotionalTone: string;
  improvementSuggestions: string[];
  additionalComments: string;
  timestamp: Date;
}

interface EnhancedFeedbackPanelProps {
  messageId: string;
  messageContent: string;
  onFeedbackSubmit: (feedback: FeedbackData) => void;
  onClose: () => void;
  showDetailed?: boolean;
}

const EnhancedFeedbackPanel: React.FC<EnhancedFeedbackPanelProps> = ({
  messageId,
  messageContent,
  onFeedbackSubmit,
  onClose,
  showDetailed = true
}) => {
  const [feedback, setFeedback] = useState<FeedbackData>({
    messageId,
    helpful: null,
    accuracy: 0,
    empathy: 0,
    safety: 0,
    clarity: 0,
    emotionalTone: '',
    improvementSuggestions: [],
    additionalComments: '',
    timestamp: new Date()
  });

  const [currentStep, setCurrentStep] = useState<'quick' | 'detailed' | 'comments'>('quick');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const quickFeedbackOptions = [
    { value: true, label: 'Helpful', icon: 'LIKE', color: '#10b981' },
    { value: false, label: 'Not Helpful', icon: 'DISLIKE', color: '#ef4444' }
  ];

  const emotionalToneOptions = [
    { value: 'supportive', label: 'Supportive', icon: 'HUG', color: '#10b981' },
    { value: 'professional', label: 'Professional', icon: 'TIE', color: '#3b82f6' },
    { value: 'empathetic', label: 'Empathetic', icon: 'HEART', color: '#f59e0b' },
    { value: 'caring', label: 'Caring', icon: 'HANDS', color: '#8b5cf6' },
    { value: 'cold', label: 'Cold', icon: 'ICE', color: '#6b7280' },
    { value: 'inappropriate', label: 'Inappropriate', icon: 'WARNING', color: '#ef4444' }
  ];

  const improvementSuggestionOptions = [
    'More specific techniques',
    'Better emotional support',
    'Clearer explanations',
    'More practical advice',
    'Better safety guidance',
    'More personalized response',
    'Include more resources',
    'Improve tone and empathy'
  ];

  const renderStarRating = (value: number, onChange: (value: number) => void, label: string) => (
    <div style={{ marginBottom: '16px' }}>
      <div style={{
        fontSize: '13px',
        fontWeight: '600',
        color: '#374151',
        marginBottom: '8px'
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', gap: '4px' }}>
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => onChange(star)}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              fontSize: '20px',
              padding: '2px',
              color: star <= value ? '#f59e0b' : '#d1d5db',
              transition: 'color 0.1s'
            }}
            onMouseEnter={(e) => {
              if (star > value) {
                e.currentTarget.style.color = '#fbbf24';
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = star <= value ? '#f59e0b' : '#d1d5db';
            }}
          >
            ⭐
          </button>
        ))}
        <span style={{
          marginLeft: '8px',
          fontSize: '12px',
          color: '#6b7280'
        }}>
          {value > 0 ? `${value}/5` : 'Not rated'}
        </span>
      </div>
    </div>
  );

  const renderQuickFeedback = () => (
    <div>
      <div style={{
        fontSize: '16px',
        fontWeight: '600',
        color: '#1f2937',
        marginBottom: '16px'
      }}>
        Was this response helpful?
      </div>
      
      <div style={{
        display: 'flex',
        gap: '12px',
        marginBottom: '20px'
      }}>
        {quickFeedbackOptions.map((option) => (
          <button
            key={option.label}
            onClick={() => setFeedback({ ...feedback, helpful: option.value })}
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '8px',
              padding: '16px',
              background: feedback.helpful === option.value 
                ? `${option.color}15` 
                : 'transparent',
              border: `2px solid ${feedback.helpful === option.value ? option.color : '#e5e7eb'}`,
              borderRadius: '12px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              if (feedback.helpful !== option.value) {
                e.currentTarget.style.backgroundColor = '#f9fafb';
                e.currentTarget.style.borderColor = option.color;
              }
            }}
            onMouseLeave={(e) => {
              if (feedback.helpful !== option.value) {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.borderColor = '#e5e7eb';
              }
            }}
          >
            <span style={{ fontSize: '24px' }}>{option.icon}</span>
            <span style={{
              fontSize: '14px',
              fontWeight: '600',
              color: feedback.helpful === option.value ? option.color : '#6b7280'
            }}>
              {option.label}
            </span>
          </button>
        ))}
      </div>

      <div style={{
        display: 'flex',
        gap: '8px',
        justifyContent: 'flex-end'
      }}>
        {showDetailed && (
          <button
            onClick={() => setCurrentStep('detailed')}
            style={{
              background: 'transparent',
              border: '1px solid #d1d5db',
              borderRadius: '8px',
              padding: '8px 16px',
              fontSize: '14px',
              color: '#6b7280',
              cursor: 'pointer'
            }}
          >
            Detailed Feedback
          </button>
        )}
        <button
          onClick={handleQuickSubmit}
          disabled={feedback.helpful === null}
          style={{
            background: feedback.helpful === null 
              ? '#e5e7eb' 
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: feedback.helpful === null ? '#9ca3af' : 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: feedback.helpful === null ? 'not-allowed' : 'pointer'
          }}
        >
          Submit
        </button>
      </div>
    </div>
  );

  const renderDetailedFeedback = () => (
    <div>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: '20px'
      }}>
        <button
          onClick={() => setCurrentStep('quick')}
          style={{
            background: 'transparent',
            border: 'none',
            fontSize: '16px',
            cursor: 'pointer',
            marginRight: '8px'
          }}
        >
          ←
        </button>
        <div style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#1f2937'
        }}>
          Detailed Feedback
        </div>
      </div>

      {renderStarRating(
        feedback.accuracy,
        (value) => setFeedback({ ...feedback, accuracy: value }),
        'Accuracy & Relevance'
      )}

      {renderStarRating(
        feedback.empathy,
        (value) => setFeedback({ ...feedback, empathy: value }),
        'Empathy & Understanding'
      )}

      {renderStarRating(
        feedback.safety,
        (value) => setFeedback({ ...feedback, safety: value }),
        'Safety & Appropriateness'
      )}

      {renderStarRating(
        feedback.clarity,
        (value) => setFeedback({ ...feedback, clarity: value }),
        'Clarity & Helpfulness'
      )}

      <div style={{ marginBottom: '16px' }}>
        <div style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#374151',
          marginBottom: '8px'
        }}>
          Emotional Tone
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '8px'
        }}>
          {emotionalToneOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setFeedback({ ...feedback, emotionalTone: option.value })}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px',
                background: feedback.emotionalTone === option.value 
                  ? `${option.color}15` 
                  : 'transparent',
                border: `1px solid ${feedback.emotionalTone === option.value ? option.color : '#e5e7eb'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '12px',
                transition: 'all 0.2s'
              }}
            >
              <span>{option.icon}</span>
              <span style={{
                color: feedback.emotionalTone === option.value ? option.color : '#6b7280'
              }}>
                {option.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div style={{
        display: 'flex',
        gap: '8px',
        justifyContent: 'flex-end'
      }}>
        <button
          onClick={() => setCurrentStep('comments')}
          style={{
            background: 'transparent',
            border: '1px solid #d1d5db',
            borderRadius: '8px',
            padding: '8px 16px',
            fontSize: '14px',
            color: '#6b7280',
            cursor: 'pointer'
          }}
        >
          Add Comments
        </button>
        <button
          onClick={handleDetailedSubmit}
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer'
          }}
        >
          Submit Feedback
        </button>
      </div>
    </div>
  );

  const renderComments = () => (
    <div>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: '20px'
      }}>
        <button
          onClick={() => setCurrentStep('detailed')}
          style={{
            background: 'transparent',
            border: 'none',
            fontSize: '16px',
            cursor: 'pointer',
            marginRight: '8px'
          }}
        >
          ←
        </button>
        <div style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#1f2937'
        }}>
          Additional Comments
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <div style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#374151',
          marginBottom: '8px'
        }}>
          Suggestions for Improvement
        </div>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px',
          marginBottom: '12px'
        }}>
          {improvementSuggestionOptions.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => {
                const suggestions = feedback.improvementSuggestions.includes(suggestion)
                  ? feedback.improvementSuggestions.filter(s => s !== suggestion)
                  : [...feedback.improvementSuggestions, suggestion];
                setFeedback({ ...feedback, improvementSuggestions: suggestions });
              }}
              style={{
                background: feedback.improvementSuggestions.includes(suggestion) 
                  ? 'rgba(99, 102, 241, 0.1)' 
                  : 'transparent',
                border: `1px solid ${feedback.improvementSuggestions.includes(suggestion) ? '#6366f1' : '#e5e7eb'}`,
                borderRadius: '16px',
                padding: '4px 8px',
                fontSize: '11px',
                color: feedback.improvementSuggestions.includes(suggestion) ? '#6366f1' : '#6b7280',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <div style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#374151',
          marginBottom: '8px'
        }}>
          Additional Comments
        </div>
        <textarea
          value={feedback.additionalComments}
          onChange={(e) => setFeedback({ ...feedback, additionalComments: e.target.value })}
          placeholder="Share any additional thoughts or suggestions..."
          style={{
            width: '100%',
            minHeight: '80px',
            padding: '12px',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            fontSize: '14px',
            resize: 'vertical',
            outline: 'none'
          }}
        />
      </div>

      <div style={{
        display: 'flex',
        gap: '8px',
        justifyContent: 'flex-end'
      }}>
        <button
          onClick={handleDetailedSubmit}
          disabled={isSubmitting}
          style={{
            background: isSubmitting 
              ? '#e5e7eb' 
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: isSubmitting ? '#9ca3af' : 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: isSubmitting ? 'not-allowed' : 'pointer'
          }}
        >
          {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
        </button>
      </div>
    </div>
  );

  const handleQuickSubmit = async () => {
    if (feedback.helpful === null) return;
    
    setIsSubmitting(true);
    await onFeedbackSubmit({
      ...feedback,
      timestamp: new Date()
    });
    setIsSubmitting(false);
    onClose();
  };

  const handleDetailedSubmit = async () => {
    setIsSubmitting(true);
    await onFeedbackSubmit({
      ...feedback,
      timestamp: new Date()
    });
    setIsSubmitting(false);
    onClose();
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'quick':
        return renderQuickFeedback();
      case 'detailed':
        return renderDetailedFeedback();
      case 'comments':
        return renderComments();
      default:
        return renderQuickFeedback();
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        padding: '24px',
        maxWidth: '500px',
        width: '100%',
        maxHeight: '80vh',
        overflowY: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '20px'
        }}>
          <div style={{
            fontSize: '18px',
            fontWeight: '700',
            color: '#1f2937'
          }}>
            Feedback
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '20px',
              cursor: 'pointer',
              color: '#6b7280',
              padding: '4px'
            }}
          >
            ×
          </button>
        </div>

        <div style={{
          background: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '12px',
          marginBottom: '20px',
          fontSize: '13px',
          color: '#64748b',
          maxHeight: '100px',
          overflowY: 'auto'
        }}>
          <strong>Response:</strong> {messageContent.substring(0, 200)}
          {messageContent.length > 200 && '...'}
        </div>

        {renderCurrentStep()}
      </div>
    </div>
  );
};

export default EnhancedFeedbackPanel; 