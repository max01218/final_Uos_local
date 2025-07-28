import React, { useState, useRef, useEffect } from 'react';

interface EnhancedChatInputProps {
  onSend: (message: string, metadata?: any) => void;
  onTyping?: (isTyping: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
  suggestions?: string[];
  showEmotionDetection?: boolean;
  showSafetyWarnings?: boolean;
}

const EnhancedChatInput: React.FC<EnhancedChatInputProps> = ({
  onSend,
  onTyping,
  disabled = false,
  placeholder = "Share what's on your mind...",
  suggestions = [],
  showEmotionDetection = true,
  showSafetyWarnings = true
}) => {
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState<string | null>(null);
  const [urgencyLevel, setUrgencyLevel] = useState(0);
  const [showSafetyPrompt, setShowSafetyPrompt] = useState(false);
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Default suggestions for different scenarios
  const defaultSuggestions = [
    "I'm feeling anxious and need some coping strategies",
    "What are some relaxation techniques I can try?",
    "I'm having trouble sleeping lately",
    "How do I deal with negative thoughts?",
    "I feel overwhelmed with work stress",
    "Can you help me understand my emotions better?",
    "What are some grounding techniques?",
    "I'm struggling with mood changes"
  ];

  // Safety keywords for crisis detection
  const safetyKeywords = [
    'suicide', 'kill myself', 'hurt myself', 'end it all', 
    'cant go on', 'not worth living', 'want to die', 'harm myself'
  ];

  // Emotion detection keywords
  const emotionKeywords = {
    anxiety: ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared'],
    depression: ['depressed', 'sad', 'down', 'hopeless', 'empty', 'worthless'],
    anger: ['angry', 'mad', 'furious', 'irritated', 'frustrated'],
    stress: ['stressed', 'overwhelmed', 'pressure', 'burden'],
    loneliness: ['lonely', 'alone', 'isolated', 'disconnected']
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  useEffect(() => {
    if (message.trim()) {
      analyzeMessage(message);
      filterSuggestions(message);
    } else {
      setDetectedEmotion(null);
      setUrgencyLevel(0);
      setShowSafetyPrompt(false);
      setFilteredSuggestions([]);
    }
  }, [message]);

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const analyzeMessage = (text: string) => {
    const lowercaseText = text.toLowerCase();
    
    // Safety analysis
    if (showSafetyWarnings) {
      const hasSafetyRisk = safetyKeywords.some(keyword => 
        lowercaseText.includes(keyword)
      );
      setShowSafetyPrompt(hasSafetyRisk);
      
      if (hasSafetyRisk) {
        setUrgencyLevel(5);
        return;
      }
    }

    // Emotion detection
    if (showEmotionDetection) {
      for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
        if (keywords.some(keyword => lowercaseText.includes(keyword))) {
          setDetectedEmotion(emotion);
          setUrgencyLevel(emotion === 'depression' ? 3 : 2);
          break;
        }
      }
    }

      // Urgency level based on intensity words
  const urgencyIndicators = {
    5: ['desperate', 'cant cope', 'breaking point', 'emergency'],
    4: ['severe', 'intense', 'unbearable', 'crisis'],
    3: ['very', 'really', 'extremely', 'badly'],
    2: ['quite', 'pretty', 'somewhat'],
    1: ['a little', 'slightly', 'mild']
  };

    for (const [level, indicators] of Object.entries(urgencyIndicators)) {
      if (indicators.some(indicator => lowercaseText.includes(indicator))) {
        setUrgencyLevel(Math.max(urgencyLevel, parseInt(level)));
        break;
      }
    }
  };

  const filterSuggestions = (text: string) => {
    const allSuggestions = [...suggestions, ...defaultSuggestions];
    const filtered = allSuggestions.filter(suggestion =>
      suggestion.toLowerCase().includes(text.toLowerCase()) ||
      text.toLowerCase().split(' ').some(word => 
        suggestion.toLowerCase().includes(word) && word.length > 2
      )
    );
    setFilteredSuggestions(filtered.slice(0, 5));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newMessage = e.target.value;
    setMessage(newMessage);
    
    // Handle typing indicator
    if (!isTyping && newMessage.trim()) {
      setIsTyping(true);
      onTyping?.(true);
    }

    // Clear typing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      onTyping?.(false);
    }, 1000);

    // Show suggestions when typing
    setShowSuggestions(newMessage.length > 2);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!message.trim() || disabled) return;

    const metadata = {
      detectedEmotion,
      urgencyLevel,
      timestamp: new Date(),
      safetyRisk: showSafetyPrompt
    };

    onSend(message.trim(), metadata);
    setMessage('');
    setShowSuggestions(false);
    setDetectedEmotion(null);
    setUrgencyLevel(0);
    setShowSafetyPrompt(false);
    
    if (isTyping) {
      setIsTyping(false);
      onTyping?.(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion);
    setShowSuggestions(false);
    textareaRef.current?.focus();
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      anxiety: '#f59e0b',
      depression: '#6366f1',
      anger: '#ef4444',
      stress: '#f97316',
      loneliness: '#8b5cf6'
    };
    return colors[emotion as keyof typeof colors] || '#6b7280';
  };

  const getUrgencyColor = (level: number) => {
    if (level >= 4) return '#ef4444';
    if (level >= 3) return '#f59e0b';
    if (level >= 2) return '#3b82f6';
    return '#6b7280';
  };

  const renderEmotionIndicator = () => {
    if (!detectedEmotion) return null;

    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '12px',
        color: getEmotionColor(detectedEmotion),
        fontWeight: '500'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: getEmotionColor(detectedEmotion)
        }} />
        {detectedEmotion} detected
      </div>
    );
  };

  const renderUrgencyIndicator = () => {
    if (urgencyLevel === 0) return null;

    const urgencyLabels = {
      1: 'Low',
      2: 'Moderate', 
      3: 'High',
      4: 'Very High',
      5: 'Critical'
    };

    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '12px',
        color: getUrgencyColor(urgencyLevel),
        fontWeight: '500'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: getUrgencyColor(urgencyLevel)
        }} />
        {urgencyLabels[urgencyLevel as keyof typeof urgencyLabels]} urgency
      </div>
    );
  };

  const renderSafetyPrompt = () => {
    if (!showSafetyPrompt) return null;

    return (
      <div style={{
        margin: '8px 0',
        padding: '12px',
        background: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: '8px',
        fontSize: '13px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          color: '#dc2626',
          fontWeight: '600',
          marginBottom: '6px'
        }}>
          <span style={{ marginRight: '6px' }}>WARNING</span>
          Safety Notice
        </div>
        <div style={{ color: '#7f1d1d', marginBottom: '8px' }}>
          I'm concerned about your safety. If you're having thoughts of suicide or self-harm, please reach out for immediate help.
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '4px 8px',
              fontSize: '11px',
              cursor: 'pointer'
            }}
            onClick={() => window.open('tel:988', '_blank')}
          >
            Call 988 (Crisis Lifeline)
          </button>
          <button
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '4px 8px',
              fontSize: '11px',
              cursor: 'pointer'
            }}
            onClick={() => window.open('sms:741741?body=HOME', '_blank')}
          >
            Text HOME to 741741
          </button>
        </div>
      </div>
    );
  };

  const renderSuggestions = () => {
    if (!showSuggestions || filteredSuggestions.length === 0) return null;

    return (
      <div style={{
        position: 'absolute',
        bottom: '100%',
        left: 0,
        right: 0,
        background: 'white',
        border: '1px solid #e1e8ed',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        maxHeight: '200px',
        overflowY: 'auto',
        zIndex: 10,
        marginBottom: '4px'
      }}>
        <div style={{
          padding: '8px 12px',
          fontSize: '12px',
          color: '#6b7280',
          fontWeight: '600',
          borderBottom: '1px solid #f3f4f6'
        }}>
          Suggested topics:
        </div>
        {filteredSuggestions.map((suggestion, idx) => (
          <div
            key={idx}
            onClick={() => handleSuggestionClick(suggestion)}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              fontSize: '13px',
              color: '#374151',
              borderBottom: idx < filteredSuggestions.length - 1 ? '1px solid #f3f4f6' : 'none',
              transition: 'background-color 0.1s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f8fafc'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
          >
            {suggestion}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{
      position: 'relative',
      background: 'white',
      borderRadius: '12px',
      border: '1px solid #e1e8ed',
      padding: '16px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    }}>
      {renderSafetyPrompt()}
      
      {renderSuggestions()}
      
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: '12px'
      }}>
        <div style={{ flex: 1 }}>
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={disabled}
            style={{
              width: '100%',
              minHeight: '44px',
              maxHeight: '120px',
              resize: 'none',
              border: 'none',
              outline: 'none',
              fontSize: '15px',
              lineHeight: '1.5',
              fontFamily: 'inherit',
              background: 'transparent'
            }}
          />
          
          {(detectedEmotion || urgencyLevel > 0) && (
            <div style={{
              display: 'flex',
              gap: '12px',
              marginTop: '8px',
              paddingTop: '8px',
              borderTop: '1px solid #f3f4f6'
            }}>
              {renderEmotionIndicator()}
              {renderUrgencyIndicator()}
            </div>
          )}
        </div>
        
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          style={{
            background: (!message.trim() || disabled) 
              ? '#e5e7eb' 
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: (!message.trim() || disabled) ? '#9ca3af' : 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '10px 16px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: (!message.trim() || disabled) ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
            minWidth: '80px'
          }}
          onMouseEnter={(e) => {
            if (message.trim() && !disabled) {
              e.currentTarget.style.transform = 'translateY(-1px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          {disabled ? 'Sending...' : 'Send'}
        </button>
      </div>
      
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: '8px',
        fontSize: '11px',
        color: '#9ca3af'
      }}>
        <span>
          {message.length > 0 && `${message.length} characters`}
        </span>
        <span>
          Press Enter to send, Shift+Enter for new line
        </span>
      </div>
    </div>
  );
};

export default EnhancedChatInput; 