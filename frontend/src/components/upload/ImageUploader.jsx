import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { UploadCloud, Image as ImageIcon, X } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";

export default function ImageUploader({ onUpload, isUploading }) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    }
  };

  const handleChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    }
  };

  const clearPreview = () => {
    setPreview(null);
  };

  return (
    <div className="w-full">
      <div
        className={`
          ios-card transition-all overflow-hidden
          ${dragActive ? 'ring-1 ring-purple-300' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-12 px-6"
            >
              <div className="mb-3">
                <div className="flex items-center justify-center gap-1">
                  <div className="text-[#9747FF] text-xs">◇</div>
                  <div className="text-[#FF5C87] text-xs">○</div>
                  <div className="text-[#42CEB4] text-xs">△</div>
                </div>
              </div>
              
              <h3 className="text-lg font-medium text-gray-800 mb-2 sf-pro-display">Upload your SET game image</h3>
              <p className="text-sm text-gray-600 mb-2 text-center sf-pro-text">
                Drag and drop your image here, or tap to browse
              </p>
              <p className="text-xs text-gray-500 mb-6 text-center max-w-xs sf-pro-text">
                For best results, ensure all cards are clearly visible, well-lit, and the photo is taken from directly above the game.
              </p>
              
              <input
                type="file"
                onChange={handleChange}
                accept="image/*"
                className="hidden"
                id="file-upload"
              />
              <div>
                <button
                  onClick={() => document.getElementById('file-upload').click()}
                  className="purple-button sf-pro-text"
                >
                  Select Image
                </button>
              </div>
              
              <p className="text-xs text-gray-400 mt-5 sf-pro-text">
                Supports PNG, JPG, JPEG (max 10MB)
              </p>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="relative"
            >
              <img
                src={preview}
                alt="Preview"
                className="w-full h-auto"
              />
              {!isUploading && (
                <Button
                  size="icon"
                  className="absolute top-3 right-3 bg-white/90 hover:bg-white border-0 shadow-md rounded-full w-9 h-9"
                  onClick={clearPreview}
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
              {isUploading && (
                <div className="absolute inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center">
                  <div className="w-3/4 max-w-xs bg-white/20 p-5 rounded-2xl backdrop-blur-md">
                    <Progress value={65} className="h-1.5 bg-white/30" />
                    <p className="text-white text-sm mt-3 text-center font-medium sf-pro-text">Analyzing your SET game...</p>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
