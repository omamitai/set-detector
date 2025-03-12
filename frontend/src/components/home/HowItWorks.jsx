import React from 'react';
import { Card } from "@/components/ui/card";
import { Camera, Cpu, Eye } from "lucide-react";

export default function HowItWorks() {
  const steps = [
    {
      icon: Camera,
      title: "Capture",
      description: "Take a clear photo of your SET game layout",
      color: "text-[#9747FF] bg-[#F8F2FF]"
    },
    {
      icon: Cpu,
      title: "Process",
      description: "AI identifies each card's unique features",
      color: "text-[#42CEB4] bg-[#F0FCFA]"
    },
    {
      icon: Eye,
      title: "Discover",
      description: "All valid SET combinations are highlighted",
      color: "text-[#FF5C87] bg-[#FFF2F5]"
    }
  ];

  return (
    <div className="py-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h2 className="text-xl font-medium text-gray-800 mb-2 sf-pro-display">How It Works</h2>
          <p className="text-gray-600 text-sm max-w-sm mx-auto sf-pro-text">
            Our AI-powered SET detector helps you find all valid combinations
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-5">
          {steps.map((step, index) => (
            <Card key={index} className="p-5 text-center ios-card">
              <div className={`w-12 h-12 mx-auto mb-4 rounded-full shadow-inner flex items-center justify-center ${step.color}`}>
                <step.icon className="w-5 h-5" />
              </div>
              <h3 className="font-medium text-gray-800 mb-1 text-base sf-pro-display">{step.title}</h3>
              <p className="text-gray-600 text-sm sf-pro-text">{step.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
