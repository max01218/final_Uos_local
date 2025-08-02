import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged,
  User as FirebaseUser,
  updateProfile
} from 'firebase/auth';
import { 
  getFirestore, 
  doc, 
  setDoc, 
  getDoc, 
  updateDoc, 
  collection, 
  query, 
  where, 
  getDocs,
  serverTimestamp,
  Timestamp
} from 'firebase/firestore';
import { AuthUser, LoginCredentials, RegisterCredentials, Conversation } from '@/types';

// Firebase configuration
// Replace with your actual Firebase config
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
export const auth = getAuth(app);
export const db = getFirestore(app);

// Convert Firebase User to our AuthUser type
const convertFirebaseUserToAuthUser = (firebaseUser: FirebaseUser): AuthUser => {
  return {
    id: firebaseUser.uid,
    email: firebaseUser.email || '',
    name: firebaseUser.displayName || '',
    createdAt: firebaseUser.metadata.creationTime ? new Date(firebaseUser.metadata.creationTime) : new Date(),
    lastActive: new Date(),
    isVerified: firebaseUser.emailVerified,
          preferences: {
        theme: 'light',
        language: 'en-US',
        notifications: true,
        autoSave: true,
        defaultTone: 'professional'
      }
  };
};

// User management functions
export const createUserDocument = async (user: AuthUser) => {
  try {
    const userRef = doc(db, 'users', user.id);
    await setDoc(userRef, {
      ...user,
      createdAt: serverTimestamp(),
      lastActive: serverTimestamp(),
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error creating user document:', error);
    throw new Error('Failed to create user profile');
  }
};

export const getUserDocument = async (userId: string): Promise<AuthUser | null> => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (userSnap.exists()) {
      const data = userSnap.data();
      return {
        id: data.id,
        email: data.email,
        name: data.name,
        createdAt: data.createdAt?.toDate() || new Date(),
        lastActive: data.lastActive?.toDate() || new Date(),
        isVerified: data.isVerified,
        preferences: data.preferences
      };
    }
    return null;
  } catch (error) {
    console.error('Error getting user document:', error);
    throw new Error('Failed to get user profile');
  }
};

export const updateUserDocument = async (userId: string, updates: Partial<AuthUser>) => {
  try {
    const userRef = doc(db, 'users', userId);
    await updateDoc(userRef, {
      ...updates,
      lastActive: serverTimestamp(),
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error updating user document:', error);
    throw new Error('Failed to update user profile');
  }
};

// Authentication functions
export const registerWithFirebase = async (credentials: RegisterCredentials): Promise<{ user: AuthUser; token: string; refreshToken: string }> => {
  try {
    // Create user with Firebase Auth
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      credentials.email,
      credentials.password
    );

    const firebaseUser = userCredential.user;

    // Update display name
    if (credentials.name) {
      await updateProfile(firebaseUser, {
        displayName: credentials.name
      });
    }

    // Get ID token
    const token = await firebaseUser.getIdToken();
    const refreshToken = firebaseUser.refreshToken;

    // Convert to our AuthUser type
    const authUser = convertFirebaseUserToAuthUser(firebaseUser);

    // Create user document in Firestore
    await createUserDocument(authUser);

    return {
      user: authUser,
      token,
      refreshToken: refreshToken || ''
    };
  } catch (error: any) {
    console.error('Firebase registration error:', error);
    
    // Handle specific Firebase errors
    switch (error.code) {
      case 'auth/email-already-in-use':
        throw new Error('Email already registered');
      case 'auth/invalid-email':
        throw new Error('Invalid email address');
      case 'auth/weak-password':
        throw new Error('Password is too weak');
      default:
        throw new Error('Registration failed');
    }
  }
};

export const loginWithFirebase = async (credentials: LoginCredentials): Promise<{ user: AuthUser; token: string; refreshToken: string }> => {
  try {
    // Sign in with Firebase Auth
    const userCredential = await signInWithEmailAndPassword(
      auth,
      credentials.email,
      credentials.password
    );

    const firebaseUser = userCredential.user;

    // Get ID token
    const token = await firebaseUser.getIdToken();
    const refreshToken = firebaseUser.refreshToken;

    // Get user document from Firestore
    const authUser = await getUserDocument(firebaseUser.uid);
    
    if (!authUser) {
      throw new Error('User profile not found');
    }

    // Update last active time
    await updateUserDocument(firebaseUser.uid, { lastActive: new Date() });

    return {
      user: authUser,
      token,
      refreshToken: refreshToken || ''
    };
  } catch (error: any) {
    console.error('Firebase login error:', error);
    
    // Handle specific Firebase errors
    switch (error.code) {
      case 'auth/user-not-found':
        throw new Error('User not found');
      case 'auth/wrong-password':
        throw new Error('Invalid password');
      case 'auth/invalid-email':
        throw new Error('Invalid email address');
      case 'auth/user-disabled':
        throw new Error('Account is disabled');
      default:
        throw new Error('Login failed');
    }
  }
};

export const logoutFromFirebase = async (): Promise<void> => {
  try {
    await signOut(auth);
  } catch (error) {
    console.error('Firebase logout error:', error);
    throw new Error('Logout failed');
  }
};

// Conversation management functions
export const saveConversation = async (userId: string, conversation: Conversation) => {
  try {
    const conversationRef = doc(db, 'conversations', conversation.id);
    await setDoc(conversationRef, {
      ...conversation,
      userId,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error saving conversation:', error);
    throw new Error('Failed to save conversation');
  }
};

export const getUserConversations = async (userId: string): Promise<Conversation[]> => {
  try {
    const conversationsRef = collection(db, 'conversations');
    const q = query(conversationsRef, where('userId', '==', userId));
    const querySnapshot = await getDocs(q);
    
    const conversations: Conversation[] = [];
    querySnapshot.forEach((doc) => {
      const data = doc.data();
      conversations.push({
        id: data.id,
        title: data.title,
        messages: data.messages,
        userId: data.userId,
        createdAt: data.createdAt?.toDate() || new Date(),
        updatedAt: data.updatedAt?.toDate() || new Date(),
        tone: data.tone || 'professional'
      });
    });
    
    return conversations.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
  } catch (error) {
    console.error('Error getting user conversations:', error);
    throw new Error('Failed to get conversations');
  }
};

export const updateConversation = async (conversationId: string, updates: Partial<Conversation>) => {
  try {
    const conversationRef = doc(db, 'conversations', conversationId);
    await updateDoc(conversationRef, {
      ...updates,
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error updating conversation:', error);
    throw new Error('Failed to update conversation');
  }
};

// Auth state listener
export const onAuthStateChange = (callback: (user: AuthUser | null) => void) => {
  return onAuthStateChanged(auth, async (firebaseUser) => {
    if (firebaseUser) {
      try {
        const authUser = await getUserDocument(firebaseUser.uid);
        callback(authUser);
      } catch (error) {
        console.error('Error getting user on auth state change:', error);
        callback(null);
      }
    } else {
      callback(null);
    }
  });
};

export default app; 