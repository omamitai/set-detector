export class GameSession {
  constructor(data) {
    this.id = data.id || data.session_id || Date.now().toString();
    this.original_image_url = data.original_image_url;
    this.processed_image_url = data.processed_image_url;
    this.detected_sets = data.detected_sets || [];
    this.status = data.status || 'pending';
    this.created_at = data.created_at || new Date().toISOString();
  }

  static async create(data) {
    return new GameSession(data);
  }
}
