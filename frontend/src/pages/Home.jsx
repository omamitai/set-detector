import React, { useState } from 'react';
import { GameSession } from '@/entities/GameSession';
import { UploadFile } from '@/integrations/Core';
import ImageUploader from '../components/upload/ImageUploader';
import ResultsView from '../components/results/ResultsView';
import HowItWorks from '../components/home/HowItWorks';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);
  
  const handleUpload = async (file) => {
    setIsUploading(true);
    setError(null);
    try {
      // Upload file to backend and get results directly
      const result = await UploadFile({ file });
      
      // Check if we have results
      if (!result || !result.detected_sets) {
        throw new Error('Invalid response from server');
      }
      
      // Create a new session with the results
      const session = await GameSession.create({
        session_id: result.session_id,
        original_image_url: result.original_image_url,
        processed_image_url: result.processed_image_url,
        detected_sets: result.detected_sets,
        status: 'completed'
      });
      
      setCurrentSession(session);
      
      // If no sets detected, show a friendly message but still show the processed image
      if (result.detected_sets.length === 0) {
        setError("No valid SETs found in the image. Try a different layout or make sure all cards are clearly visible.");
      }
    } catch (err) {
      // Provide more specific error messages based on the error
      if (err.message.includes('size exceeds')) {
        setError("Image is too large. Please upload an image smaller than 10MB.");
      } else if (err.message.includes('Only JPEG and PNG')) {
        setError("Invalid file format. Please upload a JPEG or PNG image.");
      } else {
        setError("We couldn't process your image. Please ensure it's a clear photo of a SET game layout.");
      }
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };
  
  return (
    <div className="min-h-screen">
      <div className="max-w-6xl mx-auto px-4 pt-8 pb-12 md:pt-12 md:pb-16">
        <div className="text-center mb-8 md:mb-12">
          <h1 className="text-2xl md:text-3xl font-semibold text-gray-800 mb-3 tracking-tight sf-pro-display">SET Game Detector</h1>
          <p className="text-base text-gray-600 max-w-md mx-auto sf-pro-text">
            Upload an image of your SET card game layout and we'll identify all valid sets for you.
          </p>
        </div>
        
        {error && (
          <Alert variant="destructive" className="mb-6 max-w-md mx-auto rounded-2xl bg-red-50 border-red-100 text-red-800">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="sf-pro-text">{error}</AlertDescription>
          </Alert>
        )}
        
        {!currentSession ? (
          <div className="max-w-md mx-auto mb-12">
            <ImageUploader 
              onUpload={handleUpload}
              isUploading={isUploading}
            />
          </div>
        ) : null}
        
        <ResultsView session={currentSession} />
        
        {!currentSession && <HowItWorks />}
      </div>
    </div>
  );
}
