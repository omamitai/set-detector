import React, { useState } from 'react';
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import { MapPin, Eye, EyeOff } from "lucide-react";

export default function SetCard({ set, index, onHighlight, isHighlighted }) {
  const [showDetails, setShowDetails] = useState(false);
  
  const getCardColor = (index) => {
    const colors = ["bg-[#F8F2FF] text-[#9747FF]", "bg-[#F0FCFA] text-[#42CEB4]", "bg-[#FFF2F5] text-[#FF5C87]"];
    return colors[index % colors.length];
  };
  
  const toggleDetails = () => {
    setShowDetails(!showDetails);
  };
  
  const handleHighlight = () => {
    if (onHighlight) {
      onHighlight(index);
    }
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className={`p-4 transition-shadow ios-card ${isHighlighted ? 'ring-2 ring-[#9747FF]' : 'hover:shadow-md'}`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-800 sf-pro-display">Set #{index + 1}</h3>
          <div className="flex gap-2">
            <button 
              onClick={toggleDetails} 
              className="text-gray-500 hover:text-gray-800"
              aria-label={showDetails ? "Hide card details" : "Show card details"}
            >
              {showDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
            <button 
              onClick={handleHighlight} 
              className="text-gray-500 hover:text-gray-800"
              aria-label="Highlight this set on image"
            >
              <MapPin className="w-4 h-4" />
            </button>
            <Badge variant="outline" className="text-xs font-normal rounded-full px-2.5 border-gray-200 sf-pro-text">3 Cards</Badge>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {set.cards.map((card, i) => (
            <div
              key={i}
              className={`aspect-[2/3] rounded-xl flex items-center justify-center shadow-sm ${getCardColor(i)}`}
            >
              <span className="text-xs font-medium sf-pro-text">{card}</span>
            </div>
          ))}
        </div>
        
        {showDetails && set.coordinates && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <h4 className="text-xs font-medium text-gray-700 mb-2">Card Coordinates</h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
              {set.coordinates.map((coord, i) => (
                <div key={i} className="text-gray-500 text-center">
                  <div className="bg-gray-100 rounded-md py-1">
                    x: {coord.x}, y: {coord.y}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
}
