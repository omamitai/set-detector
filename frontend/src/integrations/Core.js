// Set API URL based on environment or default to relative path for production
const API_BASE = process.env.REACT_APP_API_URL || '';
const API_ENDPOINT = `${API_BASE}/api`;

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
    console.log(`Uploading file to ${API_ENDPOINT}/detect_sets`);
    
    // Add timeout for long-running requests
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60-second timeout
    
    const response = await fetch(`${API_ENDPOINT}/detect_sets`, {
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

export const checkApiHealth = async () => {
  try {
    const response = await fetch(`${API_ENDPOINT}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      return {
        healthy: false,
        status: response.status,
        message: `Backend returned ${response.status}`
      };
    }
    
    const data = await response.json();
    return {
      healthy: data.status === 'healthy',
      memory: data.memory,
      status: response.status
    };
  } catch (error) {
    console.error('Health check failed:', error);
    return {
      healthy: false,
      message: error.message || 'Could not connect to server',
      error
    };
  }
};
