// Set API URL based on environment or default to localhost for development
const API_URL = process.env.REACT_APP_API_URL || 
                (window.location.hostname === 'localhost' ? 
                 'http://localhost:5000/api' : 
                 `${window.location.origin}/api`);

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
    const response = await fetch(`${API_URL}/detect_sets`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
};

// This function is kept for compatibility with the existing code structure
// The actual processing is done by the Python backend
export const InvokeLLM = async ({ prompt, response_json_schema }) => {
  console.warn('InvokeLLM is deprecated, using real ML backend instead');
  return {
    detected_sets: []
  };
};
