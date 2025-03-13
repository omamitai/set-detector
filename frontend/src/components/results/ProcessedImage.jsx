import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";

export default function ProcessedImage({ image, onReset }) {
  if (!image) return null;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Card className="overflow-hidden ios-card">
        <CardContent className="p-0 relative">
          <img
            src={image.processed_url}
            alt="SET game with highlighted sets"
            className="w-full h-auto"
          />
          
          <div className="absolute top-3 right-3 flex gap-2">
            <Badge className="bg-[#F8F2FF] text-[#9747FF] border-0 rounded-full px-3 py-1 shadow-sm text-xs font-medium">
              {image.sets_found} sets found
            </Badge>
          </div>

          {/* Floating action button for new analysis */}
          <motion.button 
            onClick={onReset}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 purple-button text-white flex items-center justify-center gap-2 px-6 shadow-xl hover:shadow-2xl"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Analyze New Image</span>
          </motion.button>
        </CardContent>
      </Card>
    </motion.div>
  );
}
