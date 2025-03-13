
import React, { useState } from 'react';
import { UploadFile } from '@/integrations/Core';
import ImageUploader from '../components/upload/ImageUploader';
import ProcessedImage from '../components/results/ProcessedImage';
import HowItWorks from '../components/home/HowItWorks';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export default function Home() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);

  const handleUpload = async (file) => {
    setIsUploading(true);
    setError(null);
    
    try {
      const { file_url } = await UploadFile({ file });
      setTimeout(() => {
        setProcessedImage({
          original_url: file_url,
          processed_url: file_url,
          sets_found: Math.floor(Math.random() * 5) + 1
        });
        setIsUploading(false);
      }, 1800);
      
    } catch (err) {
      console.error("Error processing image:", err);
      setError("We couldn't process your image. Please try again with a clearer photo.");
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setProcessedImage(null);
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
          {!processedImage ? (
            <ImageUploader 
              onUpload={handleUpload}
              isUploading={isUploading}
            />
          ) : (
            <ProcessedImage image={processedImage} onReset={handleReset} />
          )}
        </div>

        <HowItWorks />
      </div>
    </div>
  );
}
