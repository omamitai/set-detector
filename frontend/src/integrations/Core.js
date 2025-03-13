// Set API URL based on environment or default to relative path for production
const API_URL = process.env.REACT_APP_API_URL || '/api';

export const UploadFile = async ({ file }) => {
  // Validate file size - 10MB limit
  if (file.size > 10 * 1024 * 1024) {
    throw new Error('File size exceeds 10MB limit');
  }
  
  // Validate file type
  if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
    throw new Error('Only JPEG and PNG images are supported');
  }
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    console.log(`Uploading file to ${API_URL}/detect_sets`);
    
    // Add timeout for long-running requests
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60-second timeout
    
    const response = await fetch(`${API_URL}/detect_sets`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });
    
    // Clear the timeout
    clearTimeout(timeoutId);

    if (!response.ok) {
      // Try to parse error response
      let errorMessage;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error;
      } catch (e) {
        errorMessage = `Server error: ${response.status}`;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. The server took too long to process your image.');
    }
    console.error('Upload failed:', error);
    throw error;
  }
};
