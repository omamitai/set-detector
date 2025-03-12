import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import SetCard from './SetCard';
import { ArrowLeft } from "lucide-react";

export default function ResultsView({ session }) {
  if (!session || session.status !== 'completed') return null;

  return (
    <div className="py-8">
      <h2 className="text-xl font-medium text-gray-800 mb-8 text-center sf-pro-display">Analysis Results</h2>
      
      <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
        <Card className="overflow-hidden ios-card">
          <CardContent className="p-0 relative">
            <img
              src={session.original_image_url}
              alt="Original SET game"
              className="w-full h-auto opacity-25"
            />
            <img
              src={session.processed_image_url}
              alt="Processed SET game with highlighted sets"
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute top-3 left-3">
              <Badge className="bg-[#F8F2FF] text-[#9747FF] border-0 rounded-full px-3 shadow-sm text-xs font-normal sf-pro-text">
                Processed Image
              </Badge>
            </div>
          </CardContent>
        </Card>
        
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-medium text-gray-800 sf-pro-display">
              Found Sets
            </h3>
            <Badge className="bg-[#F0FCFA] text-[#42CEB4] border-0 rounded-full px-3 shadow-sm text-xs font-normal sf-pro-text">
              {session.detected_sets.length} valid combinations
            </Badge>
          </div>
          
          <ScrollArea className="h-[450px] pr-2">
            <div className="space-y-3">
              {session.detected_sets.map((set, index) => (
                <SetCard key={index} set={set} index={index} />
              ))}
            </div>
          </ScrollArea>
          
          <div className="mt-5 text-sm text-gray-600 bg-[#F9F9FC] p-4 rounded-xl sf-pro-text">
            <p>A valid SET contains three cards where each feature is either all the same or all different across all cards.</p>
          </div>
          
          <div className="mt-6 flex justify-center">
            <button 
              onClick={() => window.location.reload()}
              className="flex items-center gap-2 text-sm text-gray-600 bg-white py-2 px-4 rounded-full shadow-sm hover:shadow sf-pro-text transition-all"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Analyze another image</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
