import React from 'react';
import { Card } from "@/components/ui/card";
import { Camera, Cpu, Eye } from "lucide-react";

export default function HowItWorks() {
  const steps = [
    {
      icon: Camera,
      title: "Capture",
      description: "Take a clear, overhead photo with good lighting and all cards visible",
      color: "text-[#9747FF] bg-[#F8F2FF]"
    },
    {
      icon: Cpu,
      title: "Process",
      description: "Our AI algorithms identify each card's color, shape, number, and pattern",
      color: "text-[#42CEB4] bg-[#F0FCFA]"
    },
    {
      icon: Eye,
      title: "Discover",
      description: "View all valid SETs where features are either all same or all different",
      color: "text-[#FF5C87] bg-[#FFF2F5]"
    }
  ];

  return (
    <div className="pt-4 pb-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-6">
          <h2 className="text-lg font-medium text-gray-800 mb-1 sf-pro-display">How It Works</h2>
          <p className="text-gray-600 text-xs max-w-sm mx-auto sf-pro-text">
            Instantly detect all valid SET combinations in three simple steps
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 md:gap-4">
          {steps.map((step, index) => (
            <Card key={index} className="p-4 md:p-5 text-center ios-card">
              <div className={`w-11 h-11 mx-auto mb-3 rounded-full shadow-inner flex items-center justify-center ${step.color}`}>
                <step.icon className="w-5 h-5" />
              </div>
              <h3 className="font-medium text-gray-800 mb-2 text-base sf-pro-display">{step.title}</h3>
              <p className="text-gray-600 text-xs leading-relaxed sf-pro-text">{step.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
