import React, { useState, useEffect } from 'react';
import { UploadFile, checkApiHealth } from '@/integrations/Core';
import ImageUploader from '../components/upload/ImageUploader';
import ResultsView from '../components/results/ResultsView';
import HowItWorks from '../components/home/HowItWorks';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, WifiOff } from "lucide-react";
import { GameSession } from '../entities/GameSession';

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [session, setSession] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ healthy: true });

  // Check backend health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      const status = await checkApiHealth();
      setBackendStatus(status);
    };
    
    checkHealth();
    
    // Periodically check health
    const interval = setInterval(checkHealth, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async (file) => {
    setIsUploading(true);
    setError(null);
    
    // Check backend health before attempting upload
    if (!backendStatus.healthy) {
      setError("The server is currently unavailable. Please try again later.");
      setIsUploading(false);
      return;
    }
    
    try {
      // Call the actual API
      const response = await UploadFile({ file });
      
      // Validate response structure
      if (!response || !response.session_id) {
        throw new Error('Invalid response from server');
      }
      
      // Check if sets were found
      if (response.detected_sets && response.detected_sets.length === 0) {
        // Not treating this as an error, but could show a specific message
        console.log('No sets found in the image');
      }
      
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
      
      // More specific error messages based on error type
      if (err.message.includes('timeout') || err.message.includes('timed out')) {
        setError("The server took too long to process your image. Try a smaller or clearer photo.");
      } else if (err.message.includes('size')) {
        setError("Your image exceeds the 10MB size limit. Please resize it and try again.");
      } else if (err.message.includes('type') || err.message.includes('supported')) {
        setError("Only JPEG and PNG images are supported. Please select a valid image.");
      } else {
        setError(err.message || "We couldn't process your image. Please try again with a clearer photo.");
      }
    } finally {
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setSession(null);
    setError(null);
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

        {!backendStatus.healthy && (
          <Alert variant="destructive" className="mb-6 max-w-md mx-auto rounded-xl bg-amber-50 border-amber-100 text-amber-800">
            <WifiOff className="h-4 w-4" />
            <AlertDescription className="sf-pro-text">
              Server connection issues detected. Some features may be unavailable.
            </AlertDescription>
          </Alert>
        )}

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
              disabled={!backendStatus.healthy}
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
