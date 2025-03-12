import React from 'react';
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";

export default function SetCard({ set, index }) {
  const getCardColor = (index) => {
    const colors = ["bg-[#F8F2FF] text-[#9747FF]", "bg-[#F0FCFA] text-[#42CEB4]", "bg-[#FFF2F5] text-[#FF5C87]"];
    return colors[index % colors.length];
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className="p-4 hover:shadow-md transition-shadow ios-card">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-800 sf-pro-display">Set #{index + 1}</h3>
          <Badge variant="outline" className="text-xs font-normal rounded-full px-2.5 border-gray-200 sf-pro-text">3 Cards</Badge>
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
      </Card>
    </motion.div>
  );
}
