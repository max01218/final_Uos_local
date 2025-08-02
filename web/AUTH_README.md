# Authentication System

This document describes the user authentication system implemented for the ICD-11 Mental Health Assistant web application.

## Features

### User Registration
- Full name, email, and password registration
- Password confirmation validation
- Terms and conditions agreement
- Email format validation
- Password strength requirements (minimum 8 characters, uppercase, lowercase, number)

### User Login
- Email and password authentication
- Remember me functionality
- Form validation and error handling
- Secure token-based authentication

### User Management
- User profile display with avatar
- User menu with logout functionality
- Session management
- Automatic token refresh

### Security Features
- Protected routes for authenticated users
- Automatic redirect to login for unauthenticated users
- Secure token storage in localStorage
- Password hashing (mock implementation - replace with bcrypt in production)

## File Structure

```
web/src/
├── components/auth/
│   ├── AuthPage.tsx          # Main authentication page
│   ├── LoginForm.tsx         # Login form component
│   ├── RegisterForm.tsx      # Registration form component
│   ├── UserMenu.tsx          # User dropdown menu
│   └── ProtectedRoute.tsx    # Route protection component
├── lib/
│   ├── auth.ts               # Authentication utilities
│   └── AuthContext.tsx       # React context for auth state
├── pages/api/auth/
│   ├── login.ts              # Login API endpoint
│   ├── register.ts           # Registration API endpoint
│   └── logout.ts             # Logout API endpoint
├── pages/
│   ├── auth.tsx              # Authentication page route
│   ├── chat.tsx              # Protected chat page
│   └── index.tsx             # Home page with auth status
└── types/
    └── index.ts              # Authentication type definitions
```

## Usage

### For Users

1. **Registration**: Visit `/auth` and click "Sign up" to create a new account
2. **Login**: Use your email and password to sign in
3. **Access Chat**: Once authenticated, you can access the chat functionality
4. **User Menu**: Click your avatar in the header to access profile options and logout

### For Developers

#### Adding Authentication to New Pages

```tsx
import ProtectedRoute from '@/components/auth/ProtectedRoute';

export default function MyProtectedPage() {
  return (
    <ProtectedRoute>
      <div>This content is only visible to authenticated users</div>
    </ProtectedRoute>
  );
}
```

#### Using Authentication Context

```tsx
import { useAuth } from '@/lib/AuthContext';

function MyComponent() {
  const { user, isAuthenticated, login, logout } = useAuth();
  
  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }
  
  return <div>Welcome, {user?.name}!</div>;
}
```

#### API Integration

The authentication system includes API endpoints for:
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

## Implementation Notes

### Current Implementation
- Uses mock database (in-memory array) for user storage
- Simple base64 encoding for password hashing (NOT secure for production)
- Mock JWT tokens (base64 encoded strings)

### Production Considerations
1. **Database**: Replace mock database with real database (PostgreSQL, MongoDB, etc.)
2. **Password Hashing**: Use bcrypt or similar for secure password hashing
3. **JWT Tokens**: Implement proper JWT with secret keys and expiration
4. **Email Verification**: Add email verification functionality
5. **Password Reset**: Implement password reset functionality
6. **Rate Limiting**: Add rate limiting to prevent brute force attacks
7. **HTTPS**: Ensure all communication is over HTTPS
8. **Session Management**: Implement proper session management
9. **Logging**: Add authentication event logging
10. **Security Headers**: Add security headers to prevent common attacks

### Environment Variables
Add these environment variables for production:

```env
JWT_SECRET=your-secret-key-here
DATABASE_URL=your-database-connection-string
EMAIL_SERVICE_API_KEY=your-email-service-key
```

## Styling

The authentication system uses the existing design system with:
- Consistent color scheme (primary, secondary, mental health colors)
- Responsive design for mobile and desktop
- Smooth animations and transitions
- Accessible form elements
- Error states and validation feedback

## Testing

To test the authentication system:

1. Start the development server: `npm run dev`
2. Visit `http://localhost:3000`
3. Click "Get Started" to go to the auth page
4. Try registering a new account
5. Test login with the created account
6. Verify that protected routes redirect to login when not authenticated
7. Test the user menu and logout functionality

## Future Enhancements

- [ ] Email verification
- [ ] Password reset functionality
- [ ] Social login (Google, Facebook, etc.)
- [ ] Two-factor authentication
- [ ] User profile management
- [ ] Account deletion
- [ ] Session timeout handling
- [ ] Remember me with secure tokens
- [ ] Admin panel for user management 