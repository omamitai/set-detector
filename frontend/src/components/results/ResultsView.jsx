import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Eye } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import SetCard from './SetCard';

export default function ResultsView({ session, onReset }) {
  if (!session || session.status !== 'completed') return null;

  // Ensure detected_sets exists and is an array
  const detectedSets = Array.isArray(session.detected_sets) ? session.detected_sets : [];

  return (
    <div className="py-4 mb-6">
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="overflow-hidden ios-card">
          <CardContent className="p-0 relative">
            <img
              src={session.processed_image_url}
              alt="SET game with highlighted sets"
              className="w-full h-auto"
            />
            
            <div className="absolute top-3 right-3 flex gap-2">
              <Badge className="bg-[#F8F2FF] text-[#9747FF] border-0 rounded-full px-3 py-1 shadow-sm text-xs font-normal">
                {detectedSets.length} sets found
              </Badge>
            </div>
          </CardContent>
        </Card>
        
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-medium text-gray-800 sf-pro-display flex items-center gap-1.5">
              <Eye className="w-4 h-4 text-[#9747FF]" />
              Found Sets
            </h3>
            <Badge className="bg-[#F0FCFA] text-[#42CEB4] border-0 rounded-full px-3 shadow-sm text-xs font-normal">
              {detectedSets.length} combinations
            </Badge>
          </div>
          
          <ScrollArea className="h-[300px] pr-2">
            <div className="space-y-3">
              {detectedSets.map((set, index) => (
                <SetCard key={index} set={set} index={index} />
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>
      
      <div className="flex justify-center mt-8">
        <button 
          onClick={onReset}
          className="purple-button text-white flex items-center justify-center gap-2 px-6"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Analyze Another Image</span>
        </button>
      </div>
    </div>
  );
}
