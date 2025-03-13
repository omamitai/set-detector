/**
 * Shared configuration for file uploads
 * This ensures consistent validation between frontend and backend
 */
export const UPLOAD_CONFIG = {
  // File size limits
  maxSizeMB: 10,
  maxSizeBytes: 10 * 1024 * 1024, // 10MB in bytes
  
  // Accepted file types
  allowedTypes: ['image/jpeg', 'image/png', 'image/jpg'],
  allowedExtensions: ['jpg', 'jpeg', 'png'],
  
  // Error messages for consistent user feedback
  errorMessages: {
    size: 'File size exceeds 10MB limit. Please resize your image and try again.',
    type: 'Only JPEG and PNG images are supported. Please select a valid image file.',
    noFile: 'Please select an image file to upload.',
    timeout: 'The server took too long to process your image. Try a smaller or clearer photo.',
    generic: 'Could not process your image. Please try using a clearer photo with good lighting.'
  },
  
  // Processing settings
  processingTimeout: 60000, // 60 seconds timeout for processing
  
  // Upload instructions for users
  uploadInstructions: [
    'Take a clear photo with good lighting',
    'Ensure all SET cards are fully visible',
    'Hold the camera directly above the cards',
    'Avoid shadows and reflections on cards',
    'Make sure card colors are clearly visible'
  ]
};

export default UPLOAD_CONFIG;
