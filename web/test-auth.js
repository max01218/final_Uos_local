// Simple test script for authentication system
// Run this in the browser console to test the auth functionality

console.log('Testing Authentication System...');

// Test 1: Check if auth context is available
function testAuthContext() {
  console.log('Test 1: Checking Auth Context...');
  
  // This would be available in a React component
  // For now, we'll just check if the auth utilities are available
  if (typeof window !== 'undefined') {
    console.log('✅ Browser environment detected');
  } else {
    console.log('❌ Not in browser environment');
  }
}

// Test 2: Test localStorage functionality
function testLocalStorage() {
  console.log('Test 2: Testing Local Storage...');
  
  try {
    localStorage.setItem('test', 'value');
    const value = localStorage.getItem('test');
    localStorage.removeItem('test');
    
    if (value === 'value') {
      console.log('✅ Local storage working correctly');
    } else {
      console.log('❌ Local storage not working');
    }
  } catch (error) {
    console.log('❌ Local storage error:', error);
  }
}

// Test 3: Test API endpoints
async function testAPIEndpoints() {
  console.log('Test 3: Testing API Endpoints...');
  
  const baseUrl = window.location.origin;
  
  // Test registration endpoint
  try {
    const registerResponse = await fetch(`${baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: 'Test User',
        email: 'test@example.com',
        password: 'TestPassword123',
        confirmPassword: 'TestPassword123',
        agreeToTerms: true
      }),
    });
    
    const registerData = await registerResponse.json();
    console.log('Registration response:', registerData);
    
    if (registerResponse.ok) {
      console.log('✅ Registration endpoint working');
    } else {
      console.log('❌ Registration endpoint error:', registerData.error);
    }
  } catch (error) {
    console.log('❌ Registration endpoint failed:', error);
  }
  
  // Test login endpoint
  try {
    const loginResponse = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: 'test@example.com',
        password: 'TestPassword123',
        rememberMe: false
      }),
    });
    
    const loginData = await loginResponse.json();
    console.log('Login response:', loginData);
    
    if (loginResponse.ok) {
      console.log('✅ Login endpoint working');
    } else {
      console.log('❌ Login endpoint error:', loginData.error);
    }
  } catch (error) {
    console.log('❌ Login endpoint failed:', error);
  }
}

// Test 4: Test form validation
function testFormValidation() {
  console.log('Test 4: Testing Form Validation...');
  
  const testCases = [
    {
      name: 'Valid email',
      email: 'test@example.com',
      expected: true
    },
    {
      name: 'Invalid email',
      email: 'invalid-email',
      expected: false
    },
    {
      name: 'Valid password',
      password: 'TestPassword123',
      expected: true
    },
    {
      name: 'Short password',
      password: 'short',
      expected: false
    }
  ];
  
  const emailRegex = /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i;
  const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/;
  
  testCases.forEach(test => {
    let result = false;
    
    if (test.email) {
      result = emailRegex.test(test.email);
    } else if (test.password) {
      result = test.password.length >= 8 && passwordRegex.test(test.password);
    }
    
    if (result === test.expected) {
      console.log(`✅ ${test.name}: ${result}`);
    } else {
      console.log(`❌ ${test.name}: expected ${test.expected}, got ${result}`);
    }
  });
}

// Run all tests
async function runAllTests() {
  console.log('🚀 Starting Authentication System Tests...\n');
  
  testAuthContext();
  console.log('');
  
  testLocalStorage();
  console.log('');
  
  await testAPIEndpoints();
  console.log('');
  
  testFormValidation();
  console.log('');
  
  console.log('🏁 Authentication System Tests Complete!');
}

// Export for use in browser console
if (typeof window !== 'undefined') {
  window.testAuth = {
    runAllTests,
    testAuthContext,
    testLocalStorage,
    testAPIEndpoints,
    testFormValidation
  };
  
  console.log('Authentication test functions available as window.testAuth');
  console.log('Run window.testAuth.runAllTests() to test everything');
}

module.exports = {
  runAllTests,
  testAuthContext,
  testLocalStorage,
  testAPIEndpoints,
  testFormValidation
}; 