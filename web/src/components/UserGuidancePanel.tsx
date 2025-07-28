import React, { useState } from 'react';

interface UserGuidancePanelProps {
  onClose: () => void;
  onStartChat: (message: string) => void;
  isFirstTime?: boolean;
}

const UserGuidancePanel: React.FC<UserGuidancePanelProps> = ({
  onClose,
  onStartChat,
  isFirstTime = false
}) => {
  const [currentTab, setCurrentTab] = useState<'welcome' | 'capabilities' | 'examples' | 'safety'>('welcome');

  const tabs = [
    { id: 'welcome', label: 'Welcome', icon: 'WELCOME' },
    { id: 'capabilities', label: 'What I Can Do', icon: 'BRAIN' },
    { id: 'examples', label: 'Examples', icon: 'CHAT' },
    { id: 'safety', label: 'Safety & Privacy', icon: 'SHIELD' }
  ];

  const exampleConversations = [
    {
      category: 'Anxiety Support',
      icon: 'ANXIETY',
      color: '#f59e0b',
      examples: [
        "I'm feeling really anxious about an upcoming presentation",
        "My anxiety is making it hard to sleep at night",
        "What are some quick techniques to calm my nerves?"
      ]
    },
    {
      category: 'Depression & Mood',
      icon: 'DEPRESSION',
      color: '#6366f1',
      examples: [
        "I've been feeling down and unmotivated lately",
        "Everything feels overwhelming and I don't know where to start",
        "How can I improve my mood when I'm feeling hopeless?"
      ]
    },
    {
      category: 'Stress Management',
      icon: 'STRESS',
      color: '#ef4444',
      examples: [
        "Work stress is affecting my personal life",
        "I feel burned out and need some coping strategies",
        "How do I manage stress without it overwhelming me?"
      ]
    },
    {
      category: 'Relationship Issues',
      icon: 'RELATIONSHIP',
      color: '#8b5cf6',
      examples: [
        "I'm having communication problems with my partner",
        "I feel lonely and isolated from others",
        "How do I set healthy boundaries in relationships?"
      ]
    },
    {
      category: 'Self-Care & Wellness',
      icon: 'WELLNESS',
      color: '#10b981',
      examples: [
        "What are some daily habits for better mental health?",
        "I want to develop more self-compassion",
        "How can I build resilience and emotional strength?"
      ]
    }
  ];

  const systemCapabilities = [
    {
      title: 'Intelligent Response System',
      description: 'AI combines medical knowledge (ICD-11) with therapeutic techniques (CBT) for comprehensive support',
      icon: 'BRAIN',
      features: [
        'Evidence-based therapeutic approaches',
        'Medical context and understanding',
        'Personalized response strategies'
      ]
    },
    {
      title: 'Emotional Intelligence',
      description: 'Advanced emotion detection and appropriate response matching',
      icon: 'HEART',
      features: [
        'Emotion and urgency detection',
        'Empathetic and professional tone',
        'Context-aware conversations'
      ]
    },
    {
      title: 'Safety Monitoring',
      description: 'Proactive crisis detection with immediate resource provision',
      icon: 'ALERT',
      features: [
        'Crisis keyword detection',
        'Immediate safety resources',
        'Professional help referrals'
      ]
    },
    {
      title: 'Practical Tools',
      description: 'CBT techniques and coping strategies you can use immediately',
      icon: 'TOOLS',
      features: [
        'Breathing and grounding exercises',
        'Cognitive restructuring techniques',
        'Behavioral activation strategies'
      ]
    }
  ];

  const safetyGuidelines = [
    {
      icon: 'ALERT',
      title: 'Crisis Support',
      content: 'If you\'re having thoughts of self-harm or suicide, please contact emergency services immediately or call 988 (Suicide & Crisis Lifeline).'
    },
    {
      icon: 'HOSPITAL',
      title: 'Not a Replacement for Professional Care',
      content: 'This AI assistant provides support and information but cannot replace professional mental health treatment, therapy, or medical advice.'
    },
    {
      icon: 'LOCK',
      title: 'Privacy & Confidentiality',
      content: 'Your conversations are used to improve responses. Avoid sharing personally identifiable information like names, addresses, or sensitive details.'
    },
    {
      icon: 'MEDICINE',
      title: 'Medical Disclaimer',
      content: 'This system cannot diagnose conditions or recommend medications. For medical concerns, please consult qualified healthcare professionals.'
    },
    {
      icon: 'HANDSHAKE',
      title: 'How to Get the Best Support',
      content: 'Be specific about your feelings and situations. The more context you provide, the more helpful and relevant the guidance can be.'
    }
  ];

  const handleExampleClick = (example: string) => {
    onStartChat(example);
    onClose();
  };

  const renderWelcome = () => (
    <div>
      <div style={{
        textAlign: 'center',
        marginBottom: '24px'
      }}>
        <div style={{
          fontSize: '48px',
          marginBottom: '16px'
        }}>
          BRAIN HEART
        </div>
        <h2 style={{
          fontSize: '24px',
          fontWeight: '700',
          color: '#1f2937',
          marginBottom: '8px'
        }}>
          Welcome to Your Mental Health Support Assistant
        </h2>
        <p style={{
          fontSize: '16px',
          color: '#6b7280',
          lineHeight: '1.6'
        }}>
          An intelligent AI that combines medical knowledge with therapeutic techniques to provide personalized mental health support
        </p>
      </div>

      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '12px',
        padding: '20px',
        color: 'white',
        marginBottom: '24px'
      }}>
        <h3 style={{
          fontSize: '18px',
          fontWeight: '600',
          marginBottom: '12px'
        }}>
          STAR What Makes This Different?
        </h3>
        <ul style={{
          listStyle: 'none',
          padding: 0,
          margin: 0
        }}>
          <li style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '8px' }}>SCIENCE</span>
            Evidence-based responses using ICD-11 medical knowledge
          </li>
          <li style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '8px' }}>MEDITATION</span>
            Practical CBT techniques you can use right away
          </li>
          <li style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '8px' }}>ROBOT</span>
            Smart emotion detection and personalized support
          </li>
          <li style={{ display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '8px' }}>SHIELD</span>
            Built-in safety monitoring and crisis support
          </li>
        </ul>
      </div>

      {isFirstTime && (
        <div style={{
          background: '#fef3c7',
          border: '1px solid #f59e0b',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: '8px'
          }}>
                      <span style={{ marginRight: '8px', fontSize: '16px' }}>LIGHTBULB</span>
          <strong style={{ color: '#92400e' }}>First Time Here?</strong>
          </div>
          <p style={{
            color: '#92400e',
            fontSize: '14px',
            margin: 0
          }}>
            Take a moment to explore the "What I Can Do" and "Examples" tabs to understand how to get the most from our conversation.
          </p>
        </div>
      )}

      <div style={{
        display: 'flex',
        gap: '12px',
        justifyContent: 'center'
      }}>
        <button
          onClick={() => setCurrentTab('examples')}
          style={{
            background: 'transparent',
            border: '2px solid #667eea',
            borderRadius: '8px',
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#667eea',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#667eea';
            e.currentTarget.style.color = 'white';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = '#667eea';
          }}
        >
          See Examples
        </button>
        <button
          onClick={onClose}
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer'
          }}
        >
          Start Chatting
        </button>
      </div>
    </div>
  );

  const renderCapabilities = () => (
    <div>
      <h3 style={{
        fontSize: '20px',
        fontWeight: '700',
        color: '#1f2937',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        🧠 What I Can Help You With
      </h3>

      <div style={{
        display: 'grid',
        gap: '16px',
        marginBottom: '24px'
      }}>
        {systemCapabilities.map((capability, idx) => (
          <div
            key={idx}
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              padding: '16px',
              background: 'white'
            }}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '12px'
            }}>
              <span style={{
                fontSize: '24px',
                marginRight: '12px'
              }}>
                {capability.icon}
              </span>
              <div>
                <h4 style={{
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#1f2937',
                  margin: 0
                }}>
                  {capability.title}
                </h4>
                <p style={{
                  fontSize: '14px',
                  color: '#6b7280',
                  margin: '4px 0 0 0'
                }}>
                  {capability.description}
                </p>
              </div>
            </div>
            <ul style={{
              listStyle: 'none',
              padding: 0,
              margin: 0
            }}>
              {capability.features.map((feature, featureIdx) => (
                <li
                  key={featureIdx}
                  style={{
                    fontSize: '13px',
                    color: '#4b5563',
                    marginBottom: '4px',
                    paddingLeft: '16px',
                    position: 'relative'
                  }}
                >
                  <span style={{
                    position: 'absolute',
                    left: '0',
                    color: '#10b981'
                  }}>
                    ✓
                  </span>
                  {feature}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div style={{
        background: '#f0f9ff',
        border: '1px solid #0ea5e9',
        borderRadius: '8px',
        padding: '16px',
        textAlign: 'center'
      }}>
        <p style={{
          color: '#0c4a6e',
          fontSize: '14px',
          margin: 0
        }}>
          LIGHTBULB <strong>Tip:</strong> The more specific you are about your feelings and situation, the more personalized and helpful my responses will be.
        </p>
      </div>
    </div>
  );

  const renderExamples = () => (
    <div>
      <h3 style={{
        fontSize: '20px',
        fontWeight: '700',
        color: '#1f2937',
        marginBottom: '16px',
        textAlign: 'center'
      }}>
        💬 Example Conversations
      </h3>
      
      <p style={{
        fontSize: '14px',
        color: '#6b7280',
        textAlign: 'center',
        marginBottom: '24px'
      }}>
        Click on any example to start a conversation with that topic
      </p>

      <div style={{
        display: 'grid',
        gap: '20px'
      }}>
        {exampleConversations.map((category, idx) => (
          <div
            key={idx}
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              overflow: 'hidden',
              background: 'white'
            }}
          >
            <div style={{
              background: `${category.color}10`,
              padding: '12px 16px',
              borderBottom: '1px solid #e5e7eb'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span style={{ fontSize: '20px' }}>{category.icon}</span>
                <h4 style={{
                  fontSize: '16px',
                  fontWeight: '600',
                  color: category.color,
                  margin: 0
                }}>
                  {category.category}
                </h4>
              </div>
            </div>
            
            <div style={{ padding: '16px' }}>
              {category.examples.map((example, exampleIdx) => (
                <button
                  key={exampleIdx}
                  onClick={() => handleExampleClick(example)}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    background: 'transparent',
                    border: '1px solid #f3f4f6',
                    borderRadius: '8px',
                    padding: '12px',
                    marginBottom: '8px',
                    fontSize: '14px',
                    color: '#374151',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    display: 'block'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = category.color + '05';
                    e.currentTarget.style.borderColor = category.color + '40';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = '#f3f4f6';
                  }}
                >
                  "{example}"
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderSafety = () => (
    <div>
      <h3 style={{
        fontSize: '20px',
        fontWeight: '700',
        color: '#1f2937',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        🛡️ Safety & Privacy Information
      </h3>

      <div style={{
        display: 'grid',
        gap: '16px',
        marginBottom: '24px'
      }}>
        {safetyGuidelines.map((guideline, idx) => (
          <div
            key={idx}
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              padding: '16px',
              background: 'white'
            }}
          >
            <div style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px'
            }}>
              <span style={{
                fontSize: '20px',
                marginTop: '2px'
              }}>
                {guideline.icon}
              </span>
              <div>
                <h4 style={{
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#1f2937',
                  margin: '0 0 8px 0'
                }}>
                  {guideline.title}
                </h4>
                <p style={{
                  fontSize: '14px',
                  color: '#4b5563',
                  margin: 0,
                  lineHeight: '1.5'
                }}>
                  {guideline.content}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={{
        background: '#fef2f2',
        border: '2px solid #fecaca',
        borderRadius: '12px',
        padding: '16px',
        textAlign: 'center'
      }}>
        <h4 style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#dc2626',
          marginBottom: '8px'
        }}>
          ALERT Emergency Resources
        </h4>
        <p style={{
          fontSize: '14px',
          color: '#7f1d1d',
          marginBottom: '12px'
        }}>
          If you're in immediate danger or having thoughts of self-harm:
        </p>
        <div style={{
          display: 'flex',
          gap: '8px',
          justifyContent: 'center',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => window.open('tel:988', '_blank')}
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Call 988 (Crisis Lifeline)
          </button>
          <button
            onClick={() => window.open('tel:911', '_blank')}
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Call 911 (Emergency)
          </button>
          <button
            onClick={() => window.open('sms:741741?body=HOME', '_blank')}
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Text HOME to 741741
          </button>
        </div>
      </div>
    </div>
  );

  const renderCurrentTab = () => {
    switch (currentTab) {
      case 'welcome':
        return renderWelcome();
      case 'capabilities':
        return renderCapabilities();
      case 'examples':
        return renderExamples();
      case 'safety':
        return renderSafety();
      default:
        return renderWelcome();
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
        maxWidth: '600px',
        width: '100%',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '20px 24px',
          borderBottom: '1px solid #e5e7eb'
        }}>
          <div style={{
            display: 'flex',
            gap: '16px'
          }}>
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setCurrentTab(tab.id as any)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: currentTab === tab.id 
                    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' 
                    : 'transparent',
                  color: currentTab === tab.id ? 'white' : '#6b7280',
                  border: currentTab === tab.id ? 'none' : '1px solid #e5e7eb',
                  borderRadius: '8px',
                  padding: '6px 12px',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
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

        {/* Content */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px'
        }}>
          {renderCurrentTab()}
        </div>
      </div>
    </div>
  );
};

export default UserGuidancePanel; 