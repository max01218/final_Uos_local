import React from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import AuthPage from '@/components/auth/AuthPage';
import { useAuth } from '@/lib/AuthContext';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

const Auth: NextPage = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push('/chat');
    }
  }, [isAuthenticated, isLoading, router]);

  // Don't render auth page if user is already authenticated
  if (isLoading || isAuthenticated) {
    return null;
  }

  return (
    <>
      <Head>
        <title>Sign In - ICD-11 Mental Health Assistant</title>
        <meta name="description" content="Sign in to your ICD-11 Mental Health Assistant account" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <AuthPage />
    </>
  );
};

export default Auth; 