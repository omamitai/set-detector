import React, { useState, useRef } from 'react';
import { Camera, X, Upload, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { UPLOAD_CONFIG } from '@/config/UploadConfig';

export default function ImageUploader({ onUpload, isUploading, disabled = false }) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    if (disabled || isUploading) return;
    
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    if (disabled || isUploading) return;
    
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
    if (disabled || isUploading) return;
    
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    }
  };

  const clearPreview = () => {
    if (isUploading) return;
    setPreview(null);
  };

  const handleUpload = () => {
    if (disabled || isUploading) return;
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <div
        className={`
          ios-card transition-all overflow-hidden
          ${dragActive ? 'ring-2 ring-purple-300' : ''}
          ${disabled ? 'opacity-75 pointer-events-none' : ''}
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
              className="flex flex-col items-center justify-center py-8 md:py-10 px-4 md:px-6"
            >
              <div className="mb-3">
                <motion.div 
                  className="flex items-center justify-center gap-1 bounce-animation"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ 
                    duration: 1.5, 
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                >
                  <div className="text-[#9747FF] text-sm">◇</div>
                  <div className="text-[#FF5C87] text-sm">○</div>
                  <div className="text-[#42CEB4] text-sm">△</div>
                </motion.div>
              </div>
              
              <h3 className="text-lg md:text-xl font-medium text-gray-800 mb-2 sf-pro-display">Upload SET game photo</h3>
              <p className="text-sm text-gray-600 mb-6 text-center max-w-sm sf-pro-text">
                Take a well-lit photo from directly above the cards for best results
              </p>
              
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleChange}
                accept="image/*"
                className="hidden"
                id="file-upload"
                disabled={disabled || isUploading}
              />
              
              {disabled ? (
                <div className="w-full sm:w-auto px-4 sm:px-0 mb-3">
                  <div className="bg-gray-200 text-gray-500 w-full py-3 px-5 rounded-xl flex items-center justify-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    <span>Server Unavailable</span>
                  </div>
                </div>
              ) : (
                <div className="w-full sm:w-auto px-4 sm:px-0">
                  <button
                    onClick={handleUpload}
                    className="purple-button w-full flex items-center justify-center gap-2"
                    disabled={disabled || isUploading}
                  >
                    <Camera className="w-5 h-5" />
                    <span>Upload or Take Photo</span>
                  </button>
                </div>
              )}
              
              {!disabled && (
                <div className="mt-6 px-4 w-full">
                  <div className="flex items-center justify-center gap-1.5 text-xs text-gray-500 sf-pro-text">
                    <Upload className="w-3 h-3" />
                    <span>Drag and drop image here</span>
                  </div>
                </div>
              )}
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
                <button
                  className="absolute top-3 right-3 bg-white/90 hover:bg-white border-0 shadow-md rounded-full w-10 h-10 flex items-center justify-center"
                  onClick={clearPreview}
                >
                  <X className="w-4 h-4" />
                </button>
              )}
              {isUploading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0"
                >
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-b from-purple-500/10 to-transparent"
                    animate={{
                      y: ["0%", "100%"],
                      opacity: [0.3, 0.7, 0.3]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "linear"
                    }}
                  />
                  
                  <motion.div
                    className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-purple-500/0 via-purple-500 to-purple-500/0"
                    animate={{
                      y: ["0%", "100%"]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "linear"
                    }}
                  />
                  
                  <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm rounded-full px-4 py-2 shadow-lg flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
                    <span className="text-sm font-medium text-purple-700">Processing</span>
                  </div>
                  
                  <motion.div
                    className="absolute inset-0 grid grid-cols-3 grid-rows-3 gap-2 p-4"
                    initial="hidden"
                    animate="visible"
                  >
                    {Array.from({ length: 9 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="border-2 border-transparent rounded-lg"
                        variants={{
                          hidden: { borderColor: "rgba(147, 51, 234, 0)" },
                          visible: { borderColor: "rgba(147, 51, 234, 0.3)" }
                        }}
                        transition={{
                          delay: i * 0.1,
                          duration: 0.3,
                          repeat: Infinity,
                          repeatType: "reverse"
                        }}
                      />
                    ))}
                  </motion.div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {!disabled && !isUploading && !preview && (
        <div className="mt-4 text-xs text-gray-500 sf-pro-text">
          <h4 className="font-medium mb-2">For best results:</h4>
          <ul className="list-disc pl-5 space-y-1">
            {UPLOAD_CONFIG.uploadInstructions.map((instruction, i) => (
              <li key={i}>{instruction}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
