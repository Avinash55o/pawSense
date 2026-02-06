"use client";

import React from "react";
import { Brain, PawPrint, Upload, Search } from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { Dog, ArrowRight, MessageCircle, PencilRuler, Sparkles } from "lucide-react";

export default function Home() {
  return (
    <div className="bg-background snap-y snap-mandatory h-screen overflow-y-auto">
      {/* Hero Section */}
      <section className="min-h-screen flex items-center snap-center w-full bg-gradient-to-br from-background via-primary/5 to-secondary/10">
        <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-7xl">
          <div className="flex flex-col md:flex-row items-center gap-12 lg:gap-16">
            {/* Left Content */}
            <div className="flex-1 space-y-6">
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-tight">
                Identify Any{" "}
                <span className="text-primary">
                  Dog Breed
                </span>{" "}
                with AI
              </h1>

              <p className="text-lg md:text-xl text-muted-foreground max-w-xl">
                Upload a photo and get accurate breed identification with detailed information
              </p>

              <div className="flex flex-wrap gap-4 pt-2">
                <Link
                  href="/analyzer"
                  className="bg-primary hover:bg-primary/90 text-white rounded-full px-8 py-4 font-semibold flex items-center gap-2 transition-colors shadow-lg"
                >
                  Try it now
                  <ArrowRight className="w-5 h-5" />
                </Link>
                <Link
                  href="#features"
                  className="bg-muted hover:bg-muted/80 text-foreground rounded-full px-8 py-4 font-semibold flex items-center gap-2 transition-colors"
                >
                  Learn more
                </Link>
              </div>
            </div>

            {/* Right Image */}
            <div className="flex-1 relative">
              <div className="relative rounded-2xl overflow-hidden shadow-xl border border-border">
                <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 to-transparent z-10 pointer-events-none"></div>
                <div className="w-full pb-[75%] relative">
                  <Image
                    src="/dog.jpg"
                    alt="Dog breed identification example"
                    fill
                    className="object-cover"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section
        id="features"
        className="min-h-screen flex items-center snap-center w-full py-20 bg-muted/20"
      >
        <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-7xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              What You Can Do
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Identify dog breeds, get detailed information, and ask questions about any dog
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-card rounded-xl p-8 border border-border hover:shadow-md transition-shadow">
              <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Search className="w-7 h-7 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Identify Breeds</h3>
              <p className="text-muted-foreground">
                Upload a photo and get instant breed identification with confidence scores
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-card rounded-xl p-8 border border-border hover:shadow-md transition-shadow">
              <div className="w-14 h-14 rounded-lg bg-secondary/10 flex items-center justify-center mb-4">
                <MessageCircle className="w-7 h-7 text-secondary" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Ask Questions</h3>
              <p className="text-muted-foreground">
                Ask questions about breeds and get detailed answers from our AI
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-card rounded-xl p-8 border border-border hover:shadow-md transition-shadow">
              <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Brain className="w-7 h-7 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Visual Analysis</h3>
              <p className="text-muted-foreground">
                Understand what features the AI noticed in the image
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="min-h-screen flex items-center snap-center w-full py-20 bg-gradient-to-b from-muted/30 to-background">
        <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-7xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              How It Works
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Three simple steps to identify any dog breed
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">
            {/* Connecting Lines - Hidden on mobile */}
            <div className="hidden md:block absolute top-8 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-primary via-secondary to-primary opacity-30"></div>

            {/* Step 1 */}
            <div className="text-center relative z-10">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center mx-auto mb-4">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Upload</h3>
              <p className="text-muted-foreground">
                Take or upload a photo of any dog
              </p>
            </div>

            {/* Step 2 */}
            <div className="text-center relative z-10">
              <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center mx-auto mb-4">
                <PencilRuler className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Analyze</h3>
              <p className="text-muted-foreground">
                AI processes the image to identify the breed
              </p>
            </div>

            {/* Step 3 */}
            <div className="text-center relative z-10">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center mx-auto mb-4">
                <Dog className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Discover</h3>
              <p className="text-muted-foreground">
                Get breed details and ask questions
              </p>
            </div>
          </div>

          <div className="text-center mt-12">
            <Link
              href="/analyzer"
              className="bg-primary hover:bg-primary/90 text-white rounded-full px-8 py-4 font-semibold inline-flex items-center gap-2 transition-colors"
            >
              Start Identifying
              <PawPrint className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gradient-to-b from-background to-muted/50 border-t border-border py-16 snap-end w-full">
        <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-7xl">
          <div className="flex flex-col items-center justify-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-gradient-primary flex items-center justify-center shadow-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <span className="font-bold text-2xl">PawSense</span>
            </div>
            <p className="text-center text-muted-foreground max-w-md leading-relaxed">
              AI-powered dog breed identification with visual reasoning and
              natural language capabilities
            </p>
            <Link
              href="https://avinashboruah.vercel.app/"
              className="bg-gradient-primary hover:opacity-90 text-white rounded-full px-6 py-3 font-semibold transition-smooth shadow-lg hover-lift"
            >
              Know About Author
            </Link>
            <div className="mt-4 text-sm text-muted-foreground">
              © 2025 PawSense. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
