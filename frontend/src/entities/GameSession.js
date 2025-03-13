export class GameSession {
  constructor(data) {
    this.id = data.id || data.session_id || Date.now().toString();
    this.original_image_url = data.original_image_url;
    this.processed_image_url = data.processed_image_url;
    this.detected_sets = data.detected_sets || [];
    this.status = data.status || 'pending';
    this.created_at = data.created_at || new Date().toISOString();
    this.error = data.error || null;
  }

  static async create(data) {
    // Validation to ensure data has required fields
    if (!data) {
      throw new Error('No data provided to create GameSession');
    }
    
    // Ensure detected_sets exists
    if (!data.detected_sets) {
      data.detected_sets = [];
    }
    
    // Ensure all image URLs have proper formatting
    if (data.original_image_url) {
      data.original_image_url = this.ensureCorrectImageUrl(data.original_image_url);
    }
    
    if (data.processed_image_url) {
      data.processed_image_url = this.ensureCorrectImageUrl(data.processed_image_url);
    }
    
    return new GameSession(data);
  }

  // Utility method to ensure image URLs have the correct base path
  static ensureCorrectImageUrl(url) {
    // If the URL is already absolute, return it as is
    if (url.startsWith('http')) {
      return url;
    }
    
    // If the URL already has the /api prefix, add the API_BASE
    if (url.startsWith('/api/')) {
      const API_BASE = process.env.REACT_APP_API_URL || '';
      return `${API_BASE}${url}`;
    }
    
    // Otherwise, ensure it has the /api prefix
    const API_BASE = process.env.REACT_APP_API_URL || '';
    const API_ENDPOINT = process.env.REACT_APP_API_ENDPOINT || '/api';
    return `${API_BASE}${API_ENDPOINT}${url.startsWith('/') ? url : `/${url}`}`;
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
