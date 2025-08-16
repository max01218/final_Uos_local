import React, { useEffect } from 'react';
import Head from 'next/head';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import HomeButton from '@/components/ui/HomeButton';
import { useRouter } from 'next/router';
import { toast } from 'react-hot-toast';
import { useAuth } from '@/lib/AuthContext';
import { updateUserDocument } from '@/lib/firebase';
import { useForm } from 'react-hook-form';
import { AuthUser } from '@/types';

type ProfileForm = {
  name: string;
  gender: 'male' | 'female' | 'prefer_not_to_say';
  age: number;
  occupation: string;
};

function ProfileContent() {
  const { user, updateUser } = useAuth();
  const router = useRouter();
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<ProfileForm>();

  useEffect(() => {
    if (user) {
      reset({
        name: user.name || '',
        gender: (user.gender as ProfileForm['gender']) || 'prefer_not_to_say',
        age: user.age || 18,
        occupation: user.occupation || '',
      });
    }
  }, [user, reset]);

  const onSubmit = async (data: ProfileForm) => {
    if (!user) return;
    await updateUserDocument(user.id, {
      name: data.name,
      gender: data.gender,
      age: data.age,
      occupation: data.occupation,
    } as Partial<AuthUser>);
    updateUser({
      ...(user as AuthUser),
      name: data.name,
      gender: data.gender,
      age: data.age,
      occupation: data.occupation,
    });
    toast.success('Saved');
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-secondary-50">
      <Head>
        <title>Profile</title>
        <meta name="description" content="Edit your profile" />
      </Head>

      <header className="bg-white border-b border-secondary-200 shadow-soft">
        <div className="container-responsive">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-mental-500 rounded-lg" />
              <h1 className="text-lg font-semibold text-secondary-900">Profile</h1>
            </div>
            <HomeButton variant="ghost" size="sm" style="home" className="hover:bg-secondary-100 hover:text-primary-600" />
          </div>
        </div>
      </header>

      <main className="container-responsive py-8">
        <div className="max-w-2xl mx-auto bg-white rounded-xl shadow p-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div className="space-y-2">
              <label htmlFor="name" className="block text-sm font-medium text-secondary-700">Name</label>
              <Input id="name" type="text" placeholder="Your name" error={errors.name?.message as string}
                {...register('name', { required: 'Name is required', minLength: { value: 2, message: 'Too short' } })}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label htmlFor="gender" className="block text-sm font-medium text-secondary-700">Gender</label>
                <select id="gender" className="w-full border border-secondary-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                  {...register('gender', { required: 'Gender is required' })}
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="prefer_not_to_say">Prefer not to say</option>
                </select>
                {errors.gender && <p className="text-sm text-error-600">{String(errors.gender.message || '')}</p>}
              </div>

              <div className="space-y-2">
                <label htmlFor="age" className="block text-sm font-medium text-secondary-700">Age</label>
                <Input id="age" type="number" placeholder="Age" error={errors.age?.message as string}
                  {...register('age', { required: 'Age is required', valueAsNumber: true, min: { value: 13, message: 'Min 13' }, max: { value: 120, message: 'Max 120' } })}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="occupation" className="block text-sm font-medium text-secondary-700">Occupation</label>
                <Input id="occupation" type="text" placeholder="Occupation" error={errors.occupation?.message as string}
                  {...register('occupation', { required: 'Occupation is required', minLength: { value: 2, message: 'Too short' } })}
                />
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <Button type="submit" variant="primary" size="md" loading={isSubmitting} disabled={isSubmitting}>Save</Button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default function ProfilePage() {
  return (
    <ProtectedRoute>
      <ProfileContent />
    </ProtectedRoute>
  );
}


