export class GameSession {
  constructor(data) {
    this.id = data.id || data.session_id || Date.now().toString();
    this.original_image_url = data.original_image_url || '';
    this.processed_image_url = data.processed_image_url || '';
    this.detected_sets = data.detected_sets || [];
    this.status = data.status || 'pending';
    this.created_at = data.created_at || new Date().toISOString();
    this.error = data.error || null;
  }

  static async create(data) {
    // Comprehensive validation for data structure
    if (!data) {
      throw new Error('No data provided to create GameSession');
    }
    
    // Create validated data object with defaults
    const validatedData = {
      id: data.id || data.session_id || Date.now().toString(),
      original_image_url: '',
      processed_image_url: '',
      detected_sets: [],
      status: data.status || 'pending',
      created_at: data.created_at || new Date().toISOString(),
      error: data.error || null
    };
    
    // Validate and process image URLs if present
    if (typeof data.original_image_url === 'string') {
      validatedData.original_image_url = this.ensureCorrectImageUrl(data.original_image_url);
    }
    
    if (typeof data.processed_image_url === 'string') {
      validatedData.processed_image_url = this.ensureCorrectImageUrl(data.processed_image_url);
    }
    
    // Validate detected_sets structure
    if (Array.isArray(data.detected_sets)) {
      validatedData.detected_sets = data.detected_sets.map(set => {
        const validSet = { cards: [], coordinates: [] };
        
        // Validate cards array
        if (Array.isArray(set.cards)) {
          validSet.cards = set.cards.map(card => {
            // Ensure each card is a string or convert it
            return typeof card === 'string' ? card : String(card);
          });
        }
        
        // Validate coordinates array
        if (Array.isArray(set.coordinates)) {
          validSet.coordinates = set.coordinates.map(coord => {
            // Ensure each coordinate has x and y properties
            return {
              x: typeof coord?.x === 'number' ? coord.x : 0,
              y: typeof coord?.y === 'number' ? coord.y : 0
            };
          });
        }
        
        return validSet;
      });
    }
    
    return new GameSession(validatedData);
  }

  // Utility method to ensure image URLs have the correct base path
  static ensureCorrectImageUrl(url) {
    // If URL is null or undefined, return empty string
    if (!url) return '';
    
    // If the URL is already absolute, return it as is
    if (url.startsWith('http')) {
      return url;
    }
    
    const API_BASE = process.env.REACT_APP_API_URL || '';
    
    // Ensure URL starts with a forward slash for consistency
    const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
    
    // For API-specific paths, just prefix with the API base
    if (normalizedUrl.startsWith('/api/')) {
      return `${API_BASE}${normalizedUrl}`;
    }
    
    // Otherwise, construct the URL with proper endpoint path
    const API_ENDPOINT = process.env.REACT_APP_API_ENDPOINT || '/api';
    // Remove trailing slash from API_ENDPOINT if it has one
    const cleanEndpoint = API_ENDPOINT.endsWith('/') ? API_ENDPOINT.slice(0, -1) : API_ENDPOINT;
    // Remove leading slash from URL if it has one, since we'll add it in the return
    const cleanUrl = normalizedUrl.startsWith('/') ? normalizedUrl.substring(1) : normalizedUrl;
    
    return `${API_BASE}${cleanEndpoint}/${cleanUrl}`;
  }

  // Utility methods
  get setsFound() {
    return this.detected_sets?.length || 0;
  }

  isSuccessful() {
    return this.status === 'completed' && !this.error;
  }

  hasError() {
    return !!this.error;
  }
}
