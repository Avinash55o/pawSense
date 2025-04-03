"use client";

import React from 'react';
import { Brain, PawPrint, Upload, Search } from "lucide-react";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { Dog, ArrowRight, MessageCircle, PencilRuler } from "lucide-react";

export default function Home() {
  return (
    <div className="bg-background snap-y snap-mandatory h-screen overflow-y-auto">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-secondary/10 to-background min-h-screen flex items-center snap-center w-full">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="flex flex-col md:flex-row items-center gap-12">
            <div className="flex-1 space-y-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-6 h-6 text-primary" />
                <span className="font-semibold">PawSense</span>
              </div>
              <h1 className="text-4xl md:text-6xl font-bold leading-tight">
                Identify Any <span className="text-primary">Dog Breed</span> with AI
              </h1>
              <p className="text-xl text-muted-foreground max-w-xl">
                Upload a photo and get accurate breed identification, detailed characteristics, and answers to your questions.
              </p>
              <div className="flex flex-wrap gap-4 pt-4">
                <Link href="/analyzer" className="bg-primary hover:bg-primary/90 text-white rounded-full px-6 py-3 font-medium flex items-center gap-2 transition-colors shadow-sm">
                  Try it now
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link href="#features" className="bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-full px-6 py-3 font-medium flex items-center gap-2 transition-colors shadow-sm">
                  Learn more
                </Link>
              </div>
            </div>
            <div className="flex-1 relative rounded-2xl overflow-hidden shadow-lg border border-secondary/50">
              <div className="absolute inset-0 flex items-center justify-center bg-muted z-10 opacity-0 hover:opacity-100 transition-opacity">
                <Dog className="w-36 h-36 text-primary opacity-30" />
              </div>
              <div className="w-full pb-[75%] relative">
                {/* Image placeholder with correct aspect ratio */}
                <Image
                  src="/dog.jpg"
                  alt="dog"
                  fill
                  className="object-cover"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-secondary/10 min-h-screen flex items-center snap-center w-full py-16">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">What You Can Do</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our AI-powered dog breed identification system offers multiple ways to analyze and learn about dogs.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-background rounded-xl p-6 shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <Search className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Identify Breeds</h3>
              <p className="text-muted-foreground">
                Upload any dog photo and get instant breed identification with confidence scores for the top matches.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-background rounded-xl p-6 shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <MessageCircle className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Ask Questions</h3>
              <p className="text-muted-foreground">
                Ask natural language questions about the detected breed and get detailed, informative answers.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-background rounded-xl p-6 shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Visual AI Analysis</h3>
              <p className="text-muted-foreground">
                See the visual reasoning behind each identification, understanding exactly what features the AI noticed.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="min-h-screen flex items-center snap-center w-full py-16">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Three simple steps to identify and learn about any dog breed
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
           
           
            {/* Step 1 */}
            <div className="relative text-center">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center mx-auto mb-6 relative z-10 shadow-sm">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Upload</h3>
              <p className="text-muted-foreground px-4">
                Take a photo or upload an existing image of any dog
              </p>
            </div>

            {/* Step 2 */}
            <div className="relative text-center">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center mx-auto mb-6 relative z-10 shadow-sm">
                <PencilRuler className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Analyze</h3>
              <p className="text-muted-foreground px-4">
                Our AI model processes the image to identify the breed
              </p>
            </div>

            {/* Step 3 */}
            <div className="relative text-center">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center mx-auto mb-6 relative z-10 shadow-sm">
                <Dog className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Discover</h3>
              <p className="text-muted-foreground px-4">
                Get detailed breed information and ask questions
              </p>
            </div>
           
          </div>

          <div className="text-center mt-16">
            <Link href="/analyzer" className="bg-primary hover:bg-primary/90 text-white rounded-full px-8 py-4 font-medium inline-flex items-center gap-2 transition-colors shadow-sm">
              Start Identifying
              <PawPrint className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-secondary/10 border-t border-border py-12 snap-end w-full">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="flex flex-col items-center justify-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="w-6 h-6 text-primary" />
              <span className="font-semibold text-xl">PawSense</span>
            </div>
            <p className="text-center text-muted-foreground max-w-md">
              AI-powered dog breed identification with visual reasoning and natural language capabilities
            </p>
             <Link href="https://avinashboruah.vercel.app/" className='bg-primary hover:bg-primary/90 text-white rounded-full px-4 py-2'>know about auther</Link>
            <div className="mt-4 text-sm text-muted-foreground">
              © 2025 PawSense. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

