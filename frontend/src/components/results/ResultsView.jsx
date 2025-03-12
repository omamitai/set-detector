import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";

export default function ResultsView({ session }) {
  if (!session || session.status !== 'completed') return null;

  return (
    <div className="py-8">
      <h2 className="text-xl font-medium text-gray-800 mb-8 text-center sf-pro-display">Analysis Results</h2>
      
      <div className="max-w-2xl mx-auto">
        <Card className="overflow-hidden ios-card mb-6">
          <CardContent className="p-0 relative">
            <img
              src={session.processed_image_url}
              alt="Processed SET game with highlighted sets"
              className="w-full h-auto"
            />
            <div className="absolute top-3 left-3">
              <Badge className="bg-[#F8F2FF] text-[#9747FF] border-0 rounded-full px-3 shadow-sm text-xs font-normal sf-pro-text">
                Sets Highlighted
              </Badge>
            </div>
          </CardContent>
        </Card>
        
        <div className="mt-5 text-sm text-gray-600 bg-[#F9F9FC] p-4 rounded-xl sf-pro-text text-center mx-auto max-w-md">
          <p>Each colored rectangle represents a valid SET. A SET contains three cards where each feature is either all the same or all different across all cards.</p>
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
  );
}
