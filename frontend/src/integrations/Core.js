// Set API URL based on environment or default to relative path for production
const API_BASE = process.env.REACT_APP_API_URL || '';
const API_ENDPOINT = process.env.REACT_APP_API_ENDPOINT || '/api';

// Ensure API_ENDPOINT always has the correct format (ends with /)
const getEndpointUrl = (path) => {
  const baseEndpoint = API_ENDPOINT.endsWith('/') ? API_ENDPOINT : `${API_ENDPOINT}/`;
  const cleanPath = path.startsWith('/') ? path.substring(1) : path;
  
  // If we're using relative paths (empty API_BASE), just return the endpoint + path
  if (!API_BASE) {
    return `${baseEndpoint}${cleanPath}`;
  }
  
  // Otherwise, construct the full URL
  return `${API_BASE}${baseEndpoint}${cleanPath}`;
};

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
    console.log(`Uploading file to ${getEndpointUrl('detect_sets')}`);
    
    // Add timeout for long-running requests - match with backend timeout (120s)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120-second timeout to match Gunicorn
    
    const response = await fetch(getEndpointUrl('detect_sets'), {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });
    
    // Clear the timeout
    clearTimeout(timeoutId);

    // Better error handling to properly parse backend errors
    if (!response.ok) {
      try {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server error: ${response.status}`);
      } catch (e) {
        if (e instanceof SyntaxError) {
          // JSON parsing failed
          if (response.status === 413) {
            throw new Error('File size exceeds 10MB limit');
          } else if (response.status === 503) {
            throw new Error('Server is currently busy. Please try again later.');
          }
          throw new Error(`Server error: ${response.status}`);
        }
        throw e;
      }
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. The server took too long to process your image.');
    }
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      throw new Error('Network error. Check your connection and try again.');
    }
    console.error('Upload failed:', error);
    throw error;
  }
};

export const checkApiHealth = async () => {
  try {
    const response = await fetch(getEndpointUrl('health'), {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      // Add a short timeout for health checks
      signal: AbortSignal.timeout(5000)
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
