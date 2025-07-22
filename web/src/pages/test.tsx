import React, { useState } from 'react';

interface TestResult {
  endpoint: string;
  status: number;
  response: string;
  timestamp: string;
}

export default function TestPage() {
  const [question, setQuestion] = useState('I feel sad');
  const [type, setType] = useState('concise');
  const [results, setResults] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(false);

  const endpoints = [
    { name: 'Direct LLM', path: '/api/test_llm' },
    { name: 'Simple RAG', path: '/api/test_rag_simple' },
    { name: 'Main RAG', path: '/api/rag' },
    { name: 'RAG + Psychologist', path: '/api/rag_psychologist' }
  ];

  async function testEndpoint(endpoint: string) {
    setLoading(true);
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, type }),
      });

      const data = await response.text();
      
      const result: TestResult = {
        endpoint,
        status: response.status,
        response: data,
        timestamp: new Date().toLocaleTimeString()
      };

      setResults(prev => [result, ...prev]);
    } catch (error) {
      const result: TestResult = {
        endpoint,
        status: 0,
        response: `Error: ${error}`,
        timestamp: new Date().toLocaleTimeString()
      };
      setResults(prev => [result, ...prev]);
    }
    setLoading(false);
  }

  async function testAllEndpoints() {
    setLoading(true);
    for (const endpoint of endpoints) {
      await testEndpoint(endpoint.path);
    }
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
      <h1>Backend Endpoint Testing</h1>
      
      <div style={{ marginBottom: 20 }}>
        <div style={{ marginBottom: 10 }}>
          <label>Question: </label>
          <input 
            type="text" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)}
            style={{ width: 300, marginLeft: 10 }}
          />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Type: </label>
          <select 
            value={type} 
            onChange={(e) => setType(e.target.value)}
            style={{ marginLeft: 10 }}
          >
            <option value="professional">Professional</option>
            <option value="caring">Caring</option>
            <option value="concise">Concise</option>
            <option value="casual">Casual</option>
          </select>
        </div>
        <div>
          <button 
            onClick={testAllEndpoints} 
            disabled={loading}
            style={{ marginRight: 10 }}
          >
            Test All Endpoints
          </button>
          {endpoints.map(endpoint => (
            <button 
              key={endpoint.path}
              onClick={() => testEndpoint(endpoint.path)}
              disabled={loading}
              style={{ marginRight: 10 }}
            >
              Test {endpoint.name}
            </button>
          ))}
        </div>
      </div>

      <div>
        <h2>Test Results</h2>
        {results.map((result, index) => (
          <div key={index} style={{ 
            border: '1px solid #ccc', 
            marginBottom: 10, 
            padding: 10,
            backgroundColor: result.status === 200 ? '#f0f8ff' : '#fff0f0'
          }}>
            <div style={{ fontWeight: 'bold' }}>
              {result.endpoint} - Status: {result.status} - {result.timestamp}
            </div>
            <div style={{ 
              marginTop: 5, 
              whiteSpace: 'pre-wrap', 
              fontFamily: 'monospace',
              fontSize: '12px',
              maxHeight: 200,
              overflow: 'auto'
            }}>
              {result.response}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 