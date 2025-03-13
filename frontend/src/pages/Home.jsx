import React, { useState } from 'react';
import { UploadFile } from '@/integrations/Core';
import ImageUploader from '../components/upload/ImageUploader';
import ResultsView from '../components/results/ResultsView';
import HowItWorks from '../components/home/HowItWorks';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { GameSession } from '../entities/GameSession';

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [session, setSession] = useState(null);

  const handleUpload = async (file) => {
    setIsUploading(true);
    setError(null);
    
    try {
      // Call the actual API rather than using mock data
      const response = await UploadFile({ file });
      
      // Create a proper session object
      const newSession = await GameSession.create({
        session_id: response.session_id,
        original_image_url: response.original_image_url,
        processed_image_url: response.processed_image_url,
        detected_sets: response.detected_sets,
        status: 'completed'
      });
      
      setSession(newSession);
    } catch (err) {
      console.error("Error processing image:", err);
      setError(err.message || "We couldn't process your image. Please try again with a clearer photo.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setSession(null);
  };

  return (
    <div className="min-h-screen">
      <div className="max-w-6xl mx-auto px-4 pt-4 pb-20 md:pt-8 md:pb-24">
        <div className="text-center mb-5 md:mb-8">
          <h1 className="text-xl md:text-2xl font-semibold text-gray-800 mb-2 tracking-tight sf-pro-display">SET Game Detector</h1>
          <p className="text-sm md:text-base text-gray-600 max-w-md mx-auto sf-pro-text">
            Upload a photo of your SET game layout and we'll find all valid sets
          </p>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-6 max-w-md mx-auto rounded-xl bg-red-50 border-red-100 text-red-800">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="sf-pro-text">{error}</AlertDescription>
          </Alert>
        )}

        <div className="max-w-lg mx-auto mb-12">
          {!session ? (
            <ImageUploader 
              onUpload={handleUpload}
              isUploading={isUploading}
            />
          ) : (
            <ResultsView 
              session={session} 
              onReset={handleReset} 
            />
          )}
        </div>

        <HowItWorks />
      </div>
    </div>
  );
}
