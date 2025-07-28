import React, { useState, useRef, useEffect } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  type?: 'normal' | 'safety_alert' | 'suggestion' | 'follow_up';
  metadata?: {
    confidence?: number;
    fusion_strategy?: string;
    safety_notes?: string[];
    follow_up_suggestions?: string[];
    source_breakdown?: Record<string, number>;
  };
}

interface EnhancedChatWindowProps {
  messages: Message[];
  isLoading?: boolean;
  onMessageAction?: (action: string, messageId: string) => void;
  onFollowUpClick?: (suggestion: string) => void;
}

const EnhancedChatWindow: React.FC<EnhancedChatWindowProps> = ({
  messages,
  isLoading = false,
  onMessageAction,
  onFollowUpClick
}) => {
  const [expandedMessage, setExpandedMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }).format(date);
  };

  const getMessageIcon = (role: string, type?: string) => {
    if (role === 'user') return 'USER';
    if (type === 'safety_alert') return 'ALERT';
    if (type === 'suggestion') return 'IDEA';
    return 'AI';
  };

  const getMessageStyles = (role: string, type?: string): React.CSSProperties => {
    const baseStyles: React.CSSProperties = {
      padding: '16px 20px',
      borderRadius: '18px',
      maxWidth: '85%',
      wordBreak: 'break-word',
      fontSize: '15px',
      lineHeight: '1.5',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      position: 'relative',
      transition: 'all 0.2s ease'
    };

    if (role === 'user') {
      return {
        ...baseStyles,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: '#ffffff',
        marginLeft: 'auto',
      };
    }

    if (type === 'safety_alert') {
      return {
        ...baseStyles,
        background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)',
        color: '#ffffff',
        border: '2px solid #ff4757',
      };
    }

    if (type === 'suggestion') {
      return {
        ...baseStyles,
        background: 'linear-gradient(135deg, #48cae4 0%, #0096c7 100%)',
        color: '#ffffff',
      };
    }

    return {
      ...baseStyles,
      background: '#ffffff',
      color: '#2c3e50',
      border: '1px solid #e1e8ed',
    };
  };

  const renderConfidenceIndicator = (confidence?: number) => {
    if (!confidence) return null;

    const getConfidenceColor = (conf: number) => {
      if (conf >= 0.8) return '#10b981';
      if (conf >= 0.6) return '#f59e0b';
      return '#ef4444';
    };

    const getConfidenceText = (conf: number) => {
      if (conf >= 0.8) return 'High Confidence';
      if (conf >= 0.6) return 'Medium Confidence';
      return 'Low Confidence';
    };

    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginTop: '8px',
        fontSize: '12px',
        color: '#64748b'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: getConfidenceColor(confidence),
          marginRight: '6px'
        }} />
        {getConfidenceText(confidence)} ({Math.round(confidence * 100)}%)
      </div>
    );
  };

  const renderFollowUpSuggestions = (suggestions?: string[]) => {
    if (!suggestions || suggestions.length === 0) return null;

    return (
      <div style={{ marginTop: '12px' }}>
        <div style={{
          fontSize: '13px',
          color: '#64748b',
          marginBottom: '8px',
          fontWeight: '500'
        }}>
          Suggested follow-ups:
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {suggestions.map((suggestion, idx) => (
            <button
              key={idx}
              onClick={() => onFollowUpClick?.(suggestion)}
              style={{
                background: 'rgba(99, 102, 241, 0.1)',
                border: '1px solid #e0e7ff',
                borderRadius: '16px',
                padding: '6px 12px',
                fontSize: '12px',
                color: '#6366f1',
                cursor: 'pointer',
                transition: 'all 0.2s',
                transform: 'scale(1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.2)';
                e.currentTarget.style.transform = 'scale(1.02)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderSafetyAlert = (safetyNotes?: string[]) => {
    if (!safetyNotes || safetyNotes.length === 0) return null;

    return (
      <div style={{
        marginTop: '12px',
        padding: '12px',
        background: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: '12px',
        fontSize: '13px',
        animation: 'fadeInUp 0.3s ease'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          color: '#dc2626',
          fontWeight: '600',
          marginBottom: '6px'
        }}>
          <span style={{ marginRight: '6px' }}>⚠️</span>
          Safety Notice
        </div>
        {safetyNotes.map((note, idx) => (
          <div key={idx} style={{ color: '#7f1d1d', marginBottom: '4px' }}>
            • {note}
          </div>
        ))}
      </div>
    );
  };

  const renderSourceBreakdown = (sourceBreakdown?: Record<string, number>) => {
    if (!sourceBreakdown || expandedMessage === null) return null;

    return (
      <div style={{
        marginTop: '12px',
        padding: '12px',
        background: '#f8fafc',
        borderRadius: '8px',
        fontSize: '12px',
        animation: 'fadeInDown 0.3s ease'
      }}>
        <div style={{ fontWeight: '600', marginBottom: '8px', color: '#374151' }}>
          Knowledge Sources:
        </div>
        {Object.entries(sourceBreakdown).map(([source, weight]) => (
          <div key={source} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '4px'
          }}>
            <span style={{ color: '#6b7280' }}>
              {source.toUpperCase()}
            </span>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                width: '60px',
                height: '4px',
                backgroundColor: '#e5e7eb',
                borderRadius: '2px',
                marginRight: '8px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${weight * 100}%`,
                  height: '100%',
                  backgroundColor: source === 'cbt' ? '#10b981' : source === 'icd11' ? '#3b82f6' : '#6b7280',
                  transition: 'width 0.3s ease'
                }} />
              </div>
              <span style={{ color: '#374151', fontWeight: '500' }}>
                {Math.round(weight * 100)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderMessageActions = (messageId: string, role: string) => {
    if (role === 'user') return null;

    return (
      <div style={{
        display: 'flex',
        gap: '8px',
        marginTop: '8px',
        opacity: 0.7,
        transition: 'opacity 0.2s'
      }}>
        <button
          onClick={() => onMessageAction?.('copy', messageId)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            fontSize: '12px',
            transition: 'transform 0.1s'
          }}
          title="Copy message"
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          COPY
        </button>
        <button
          onClick={() => onMessageAction?.('helpful', messageId)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            fontSize: '12px',
            transition: 'transform 0.1s'
          }}
          title="Mark as helpful"
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          LIKE
        </button>
        <button
          onClick={() => onMessageAction?.('not_helpful', messageId)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            fontSize: '12px',
            transition: 'transform 0.1s'
          }}
          title="Mark as not helpful"
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          DISLIKE
        </button>
        <button
          onClick={() => setExpandedMessage(
            expandedMessage === messageId ? null : messageId
          )}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            fontSize: '12px',
            transition: 'transform 0.1s'
          }}
          title="Show details"
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          INFO
        </button>
      </div>
    );
  };

  const renderTypingIndicator = () => (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      marginBottom: '20px',
      animation: 'fadeIn 0.3s ease'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        background: '#ffffff',
        padding: '12px 16px',
        borderRadius: '18px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        border: '1px solid #e1e8ed'
      }}>
        <span style={{ marginRight: '8px' }}>AI</span>
        <span style={{ color: '#6b7280', fontSize: '14px' }}>AI is thinking</span>
        <div style={{ marginLeft: '8px', display: 'flex', gap: '2px' }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: '4px',
                height: '4px',
                borderRadius: '50%',
                backgroundColor: '#6b7280',
                animation: `pulse 1s infinite ${i * 0.2}s`
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
      
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'linear-gradient(to bottom, #f8fafc, #e2e8f0)',
        borderRadius: '12px',
        overflow: 'hidden'
      }}>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          paddingBottom: '10px'
        }}>
          {messages.map((message) => (
            <div
              key={message.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                marginBottom: '20px',
                animation: 'fadeInUp 0.3s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px' }}>
                {message.role !== 'user' && (
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '16px',
                    marginBottom: '4px'
                  }}>
                    {getMessageIcon(message.role, message.type)}
                  </div>
                )}
                
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%'
                }}>
                  <div style={getMessageStyles(message.role, message.type)}>
                    {message.content}
                    
                    {renderConfidenceIndicator(message.metadata?.confidence)}
                    {renderSafetyAlert(message.metadata?.safety_notes)}
                    {renderFollowUpSuggestions(message.metadata?.follow_up_suggestions)}
                    {renderMessageActions(message.id, message.role)}
                  </div>
                  
                  <div style={{
                    fontSize: '11px',
                    color: '#9ca3af',
                    marginTop: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    {formatTime(message.timestamp)}
                    {message.metadata?.fusion_strategy && (
                      <span style={{
                        background: 'rgba(107, 114, 128, 0.1)',
                        padding: '2px 6px',
                        borderRadius: '8px',
                        fontSize: '10px'
                      }}>
                        {message.metadata.fusion_strategy.replace('_', ' ')}
                      </span>
                    )}
                  </div>
                  
                  {expandedMessage === message.id && renderSourceBreakdown(message.metadata?.source_breakdown)}
                </div>
                
                {message.role === 'user' && (
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '16px',
                    marginBottom: '4px'
                  }}>
                    👤
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && renderTypingIndicator()}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </>
  );
};

export default EnhancedChatWindow; 