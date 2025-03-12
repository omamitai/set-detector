import React from 'react';
import { Github, BookOpen, Moon } from "lucide-react";

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      <style>{`
        @import url('https://fonts.cdnfonts.com/css/sf-pro-display');
        
        :root {
          --set-primary: #9747FF;
          --set-secondary: #42CEB4;
          --set-accent: #FF5C87;
          font-family: 'SF Pro Display', 'SF Pro Text', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
          letter-spacing: -0.015em;
        }
        body {
          color: #333333;
          background-color: #FAFAFA;
        }
        .sf-pro-display {
          font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
          letter-spacing: -0.015em;
        }
        .sf-pro-text {
          font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
          letter-spacing: -0.01em;
        }
        .btn-ios {
          background-color: var(--set-primary);
          border-radius: 16px;
          font-weight: 500;
          padding: 8px 16px;
          box-shadow: 0 4px 10px rgba(151, 71, 255, 0.2);
          transition: all 0.15s ease;
        }
        .btn-ios:hover {
          transform: translateY(-1px);
          filter: brightness(102%);
          box-shadow: 0 6px 15px rgba(151, 71, 255, 0.25);
        }
        .btn-ios:active {
          transform: translateY(0px);
        }
        .ios-bg {
          background: #FEFEFF;
          background-attachment: fixed;
          background-image: 
            radial-gradient(at 10% 20%, rgba(151, 71, 255, 0.08) 0px, transparent 40%), 
            radial-gradient(at 90% 30%, rgba(66, 206, 180, 0.08) 0px, transparent 40%),
            radial-gradient(at 50% 80%, rgba(255, 92, 135, 0.07) 0px, transparent 35%);
        }
        .ios-card {
          border-radius: 20px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.01);
          border: none;
          backdrop-filter: blur(5px);
          background-color: rgba(255, 255, 255, 0.8);
        }
        .ios-card:hover {
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
        }
        .icon-button {
          border-radius: 100%;
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: rgba(248, 246, 255, 0.9);
          backdrop-filter: blur(5px);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
          color: #9747FF;
          transition: all 0.2s ease;
        }
        .icon-button:hover {
          background-color: rgba(255, 255, 255, 1);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
        .purple-button {
          background: linear-gradient(to bottom, #9F57FF, #8E47F5);
          color: white;
          box-shadow: 0 6px 16px rgba(151, 71, 255, 0.25);
          border-radius: 16px;
          border: none;
          padding: 10px 18px;
          font-weight: 500;
          transition: all 0.2s ease;
        }
        .purple-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 8px 20px rgba(151, 71, 255, 0.3);
        }
      `}</style>

      <header className="border-b border-gray-100 bg-white/95 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2 sf-pro-display">
            <div className="flex gap-1">
              <div className="text-[#9747FF]">◇</div>
              <div className="text-[#FF5C87]">○</div>
              <div className="text-[#42CEB4]">△</div>
            </div>
            <span className="font-medium text-gray-800 text-sm">SET Detector</span>
          </div>
          <div className="flex items-center gap-2">
            <a 
              href="https://www.setgame.com/sites/default/files/instructions/SET%20INSTRUCTIONS%20-%20ENGLISH.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-gray-600 hover:text-gray-800 transition-colors text-sm px-3 py-1.5 rounded-full hover:bg-gray-50 sf-pro-text"
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>SET Rules</span>
            </a>
            <a 
              href="https://github.com/yourusername/set-detector"
              target="_blank"
              rel="noopener noreferrer"
              className="icon-button"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      <main className="flex-1 ios-bg">
        {children}
      </main>

      <footer className="border-t border-gray-100 bg-white/95 backdrop-blur-md py-3">
        <div className="max-w-6xl mx-auto px-4 text-xs text-gray-500 sf-pro-text">
          <div className="flex justify-between items-center">
            <p>SET Game Detector • Open Source</p>
            <div className="flex items-center gap-3">
              <a 
                href="https://github.com/yourusername/set-detector"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-500 hover:text-gray-800 transition-colors"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
