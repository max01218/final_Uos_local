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
  Timestamp,
  deleteDoc,
  orderBy,
  limit,
  startAfter,
  QueryDocumentSnapshot
} from 'firebase/firestore';
import { AuthUser, LoginCredentials, RegisterCredentials, Conversation, Message } from '@/types';



// Helper function to clean message structure when reading from Firebase
export const cleanMessageFromFirebase = (msg: any) => {
  // Only include message-specific fields, not conversation-level fields
  const cleanMessage: any = {
    id: msg.id || '',
    role: msg.role || 'user',
    content: msg.content || '',
    timestamp: msg.timestamp?.toDate?.() || new Date(msg.timestamp) || new Date()
  };
  
  // Only add type if it exists and is not undefined
  if (msg.type !== undefined && msg.type !== null) {
    cleanMessage.type = msg.type;
  }
  
  // Only add metadata if it exists and has valid values
  if (msg.metadata && typeof msg.metadata === 'object') {
    const cleanMetadata = Object.fromEntries(
      Object.entries(msg.metadata).filter(([_, value]) => 
        value !== undefined && value !== null
      )
    );
    
    if (Object.keys(cleanMetadata).length > 0) {
      cleanMessage.metadata = cleanMetadata;
    }
  }
  
  return cleanMessage;
};

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

// Check if Firebase is properly configured
const isFirebaseConfigured = () => {
  const requiredKeys = [
    'apiKey',
    'authDomain', 
    'projectId',
    'storageBucket',
    'messagingSenderId',
    'appId'
  ];
  
  const missingKeys = requiredKeys.filter(key => !firebaseConfig[key as keyof typeof firebaseConfig]);
  
  if (missingKeys.length > 0) {
    console.error('Firebase configuration missing:', missingKeys);
    console.error('Please create .env.local file with Firebase configuration');
    return false;
  }
  
  return true;
};

// Log Firebase configuration status
if (!isFirebaseConfigured()) {
  console.warn('Firebase is not properly configured. Conversation history features will not work.');
}

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
    
    // Clean the conversation object to remove undefined values
    const cleanConversation = {
      id: conversation.id,
      title: conversation.title,
      messages: conversation.messages?.map((msg: any) => cleanMessageFromFirebase(msg)) || [],
      userId,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
      tone: conversation.tone || 'professional',
      isArchived: conversation.isArchived || false
    };
    

    
    await setDoc(conversationRef, cleanConversation);
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
      
      // Clean message structure when reading from Firebase
      const messages = data.messages?.map((msg: any) => cleanMessageFromFirebase(msg)) || [];
      
      conversations.push({
        id: data.id,
        title: data.title,
        messages,
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
    
    // Filter out undefined values to prevent Firebase errors
    const filteredUpdates = Object.fromEntries(
      Object.entries(updates).filter(([_, value]) => value !== undefined)
    );
    
    await updateDoc(conversationRef, {
      ...filteredUpdates,
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error updating conversation:', error);
    throw new Error('Failed to update conversation');
  }
};

// Add a specific function for adding messages to conversations
export const addMessageToConversation = async (conversationId: string, message: Message) => {
  try {

    
    const conversationRef = doc(db, 'conversations', conversationId);
    
    // Get current conversation data
    const conversationDoc = await getDoc(conversationRef);
    if (!conversationDoc.exists()) {
      throw new Error('Conversation not found');
    }
    
    const currentData = conversationDoc.data();
    const currentMessages = currentData.messages || [];
    
    // Clean existing messages to remove any undefined values and conversation-level fields
    const cleanedCurrentMessages = currentMessages.map((msg: any) => cleanMessageFromFirebase(msg));
    
    // Clean the new message object to remove undefined values and conversation-level fields
    const cleanMessage = cleanMessageFromFirebase(message);
    
    // Add new message to the array
    const updatedMessages = [...cleanedCurrentMessages, cleanMessage];
    

    
    // Update the conversation with new messages
    await updateDoc(conversationRef, {
      messages: updatedMessages,
      updatedAt: serverTimestamp()
    });
    
  } catch (error) {
    console.error('Error adding message to conversation:', error);
    throw new Error('Failed to add message to conversation');
  }
};

// Enhanced conversation management functions
export const deleteConversation = async (conversationId: string) => {
  try {
    const conversationRef = doc(db, 'conversations', conversationId);
    await deleteDoc(conversationRef);
  } catch (error) {
    console.error('Error deleting conversation:', error);
    throw new Error('Failed to delete conversation');
  }
};

export const getUserConversationsPaginated = async (
  userId: string, 
  pageSize: number = 10, 
  lastDoc?: QueryDocumentSnapshot
): Promise<{ conversations: Conversation[]; lastDoc: QueryDocumentSnapshot | null; hasMore: boolean }> => {
  try {
    // Check Firebase configuration
    if (!isFirebaseConfigured()) {
      console.warn('Firebase not configured, returning empty conversations');
      return {
        conversations: [],
        lastDoc: null,
        hasMore: false
      };
    }

    const conversationsRef = collection(db, 'conversations');
    let q = query(
      conversationsRef, 
      where('userId', '==', userId),
      orderBy('updatedAt', 'desc'),
      limit(pageSize)
    );

    if (lastDoc) {
      q = query(q, startAfter(lastDoc));
    }

    const querySnapshot = await getDocs(q);
    const conversations: Conversation[] = [];
    
    querySnapshot.forEach((doc) => {
      const data = doc.data();
      
      // Clean message structure when reading from Firebase
      const messages = data.messages?.map((msg: any) => cleanMessageFromFirebase(msg)) || [];
      
      conversations.push({
        id: data.id,
        title: data.title,
        messages,
        userId: data.userId,
        createdAt: data.createdAt?.toDate() || new Date(),
        updatedAt: data.updatedAt?.toDate() || new Date(),
        tone: data.tone || 'professional'
      });
    });

    const lastVisible = querySnapshot.docs[querySnapshot.docs.length - 1];
    const hasMore = querySnapshot.docs.length === pageSize;

    return {
      conversations,
      lastDoc: lastVisible || null,
      hasMore
    };
  } catch (error) {
    console.error('Error getting paginated conversations:', error);
    throw new Error('Failed to get conversations');
  }
};

export const searchConversations = async (userId: string, searchTerm: string): Promise<Conversation[]> => {
  try {
    const conversationsRef = collection(db, 'conversations');
    const q = query(
      conversationsRef, 
      where('userId', '==', userId),
      orderBy('updatedAt', 'desc')
    );
    
    const querySnapshot = await getDocs(q);
    const conversations: Conversation[] = [];
    
    querySnapshot.forEach((doc) => {
      const data = doc.data();
      
      // Clean message structure when reading from Firebase
      const messages = data.messages?.map((msg: any) => cleanMessageFromFirebase(msg)) || [];
      
      const conversation = {
        id: data.id,
        title: data.title,
        messages,
        userId: data.userId,
        createdAt: data.createdAt?.toDate() || new Date(),
        updatedAt: data.updatedAt?.toDate() || new Date(),
        tone: data.tone || 'professional'
      };

      // Search in title and message content
      const searchLower = searchTerm.toLowerCase();
      const titleMatch = conversation.title.toLowerCase().includes(searchLower);
      const contentMatch = conversation.messages.some((msg: any) => 
        msg.content.toLowerCase().includes(searchLower)
      );

      if (titleMatch || contentMatch) {
        conversations.push(conversation);
      }
    });

    return conversations;
  } catch (error) {
    console.error('Error searching conversations:', error);
    throw new Error('Failed to search conversations');
  }
};

export const archiveConversation = async (conversationId: string, isArchived: boolean = true) => {
  try {
    const conversationRef = doc(db, 'conversations', conversationId);
    await updateDoc(conversationRef, {
      isArchived,
      updatedAt: serverTimestamp()
    });
  } catch (error) {
    console.error('Error archiving conversation:', error);
    throw new Error('Failed to archive conversation');
  }
};

export const getConversationStats = async (userId: string) => {
  try {
    const conversationsRef = collection(db, 'conversations');
    const q = query(conversationsRef, where('userId', '==', userId));
    const querySnapshot = await getDocs(q);
    
    let totalConversations = 0;
    let totalMessages = 0;
    let archivedConversations = 0;
    const toneStats: Record<string, number> = {};

    querySnapshot.forEach((doc) => {
      const data = doc.data();
      totalConversations++;
      totalMessages += data.messages?.length || 0;
      
      if (data.isArchived) {
        archivedConversations++;
      }

      const tone = data.tone || 'professional';
      toneStats[tone] = (toneStats[tone] || 0) + 1;
    });

    return {
      totalConversations,
      totalMessages,
      archivedConversations,
      activeConversations: totalConversations - archivedConversations,
      toneStats
    };
  } catch (error) {
    console.error('Error getting conversation stats:', error);
    throw new Error('Failed to get conversation stats');
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