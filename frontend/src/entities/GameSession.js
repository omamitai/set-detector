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
    // Format detected sets data if needed
    if (data.detected_sets) {
      // Ensure each set has the required format for the SetCard component
      data.detected_sets = data.detected_sets.map(set => {
        // If the API returns cards as objects, convert them to string representations
        if (set.cards && Array.isArray(set.cards) && typeof set.cards[0] !== 'string') {
          set.cards = set.cards.map(card => 
            `${card.Count} ${card.Fill} ${card.Color} ${card.Shape}`
          );
        }
        return set;
      });
    }
    
    return new GameSession(data);
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
