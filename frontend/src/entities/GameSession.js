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
    // No need for complex transformation - the backend already provides the right format
    // Just handle the case where detected_sets might be undefined
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
    // If the URL is already absolute or starts with the correct path, return it as is
    if (url.startsWith('http') || url.startsWith('/api/')) {
      return url;
    }
    
    // Otherwise, ensure it has the /api prefix
    const API_BASE = process.env.REACT_APP_API_URL || '';
    return `${API_BASE}${url.startsWith('/') ? url : `/${url}`}`;
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
