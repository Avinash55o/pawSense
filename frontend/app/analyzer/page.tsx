"use client";

import React from "react";
import { useState } from "react";
import {
  Upload,
  Brain,
  PawPrint,
  ArrowLeft,
  Send,
  Search,
  Info,
  Loader2,
  Dog,
  Sparkles,
  CheckCircle2,
  ChevronRight,
} from "lucide-react";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../../components/ui/tabs";
import { Textarea } from "../../components/ui/textarea";
import Link from "next/link";
import { toast } from "sonner";

//type definitions
type AnalysisType = "basic" | "vlm" | "reasoning";

interface Prediction {
  breed: string;
  confidence: number;
  info?: {
    size?: string;
    energy_level?: string;
    good_with_children?: string;
    trainability?: string;
    characteristics?: string[];
    description?: string;
  };
  description?: string;
  visual_reasoning?: string;
  confidence_statement?: string;
}

interface QueryResponse {
  query: string;
  response: string;
  top_breed?: string;
  is_general_question?: boolean;
  is_visual_question?: boolean;
  processing_time?: string;
}

interface AnalysisResults {
  success?: boolean;
  predictions?: Prediction[];
  top_breed?: string;
  confidence?: number;
  confidence_statement?: string;
  visual_reasoning?: string;
  comparative_reasoning?: string;
  key_visual_features?: string[];
  queryResponse?: QueryResponse;
  caption?: string;
  colors?: string;
  detailed_appearance?: string;
  error?: string;
  processing_time?: string;
}

// Tab configuration based on analysis type
interface TabConfig {
  defaultTab: string;
  visibleTabs: string[];
}

export default function Analyzer() {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [analysisType, setAnalysisType] = useState<AnalysisType>("basic");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isVisualLoading, setIsVisualLoading] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(1);

  const getTabConfig = (): TabConfig => {
    switch (analysisType) {
      case "basic":
        return { defaultTab: "overview", visibleTabs: ["overview", "details"] };
      case "vlm":
        return { defaultTab: "queries", visibleTabs: ["queries"] };
      case "reasoning":
        return {
          defaultTab: "analysis",
          visibleTabs: ["analysis"],
        };
      default:
        return { defaultTab: "overview", visibleTabs: ["overview", "details"] };
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file: File) => {
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
      setCurrentStep(2);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setResults(null);

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("file", selectedImage);

      let endpoint = "";
      switch (analysisType) {
        case "basic":
          endpoint = "/api/classification/predict";
          break;
        case "vlm":
          endpoint = "/api/vision-language/analyze";
          formData.append("include_description", "true");
          break;
        case "reasoning":
          endpoint = "/api/vision-language/reasoning";
          break;
        default:
          endpoint = "/api/classification/predict";
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        toast.error(`Analysis failed: ${data.error}`);
        return;
      }

      // Set the results directly from the backend response
      setResults(data);
      setCurrentStep(4);
    } catch (error) {
      console.error("Error analyzing image:", error);
      toast.error("Failed to analyze image. Please try again.");
    } finally {
      setIsAnalyzing(false);
      setIsLoading(false);
    }
  };

  const submitSmartQuery = async () => {
    if (!query) return;

    // Image is strictly required now
    if (!selectedImage) {
      toast.error("Please upload an image first to ask questions!");
      return;
    }

    try {
      setIsLoading(true);
      setIsVisualLoading(true);
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("query", query);

      const response = await fetch("/api/vision-language/query", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      setResults((prev) =>
        prev
          ? {
            ...prev,
            queryResponse: data,
            ...(data.caption && { caption: data.caption }),
            ...(data.colors && { colors: data.colors }),
            ...(data.detailed_appearance && {
              detailed_appearance: data.detailed_appearance,
            }),
          }
          : data
      );

      setQuery("");
    } catch (error) {
      console.error("Error submitting query:", error);
      toast.error("Failed to process query. Please try again.");
    } finally {
      setIsLoading(false);
      setIsVisualLoading(false);
    }
  };

  const suggestedQueries = [
    "What color is this dog?",
    "How does this dog look?",
    "What size is this dog?",
    "Is this breed good with children?",
    "What is the energy level?",
    "Do dogs understand human emotions?",
    "Why do dogs bark?",
  ];

  const resetWizard = () => {
    setImagePreview("");
    setSelectedImage(null);
    setResults(null);
    setCurrentStep(1);
    setAnalysisType("basic");
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="container mx-auto max-w-7xl px-4 py-4">
        {/* Header */}
        <header className="flex items-center justify-between mb-6 animate-fade-in">
          <Link href="/">
            <Button
              variant="ghost"
              className="gap-2 hover:bg-primary/10 transition-smooth"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-primary">
              PawSense
            </h1>
            <div className="flex items-center gap-1">
              <PawPrint className="w-5 h-5 text-secondary" />
            </div>
          </div>
        </header>

        {/* Progress Steps */}
        <div className="mb-6 animate-fade-in">
          <div className="flex items-center justify-center gap-4">
            {[
              { num: 1, label: "Upload" },
              { num: 2, label: "Select Type" },
              { num: 3, label: "Analyze" },
              { num: 4, label: "Results" },
            ].map((step, idx) => (
              <React.Fragment key={step.num}>
                <div className="flex flex-col items-center gap-2">
                  <div
                    className={`
                      w-12 h-12 rounded-full flex items-center justify-center font-semibold
                      transition-smooth
                      ${currentStep >= step.num
                        ? "bg-gradient-primary text-white shadow-lg"
                        : "bg-muted text-muted-foreground"
                      }
                    `}
                  >
                    {currentStep > step.num ? (
                      <CheckCircle2 className="w-6 h-6" />
                    ) : (
                      step.num
                    )}
                  </div>
                  <span
                    className={`text-sm font-medium ${currentStep >= step.num
                      ? "text-primary"
                      : "text-muted-foreground"
                      }`}
                  >
                    {step.label}
                  </span>
                </div>
                {idx < 3 && (
                  <ChevronRight
                    className={`w-5 h-5 ${currentStep > step.num
                      ? "text-primary"
                      : "text-muted-foreground"
                      }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-5xl mx-auto">
          {/* Step 1: Upload Image */}
          {currentStep === 1 && (
            <Card className="p-8 animate-scale-in hover-lift">
              <div
                className={`
                  flex flex-col items-center gap-6 py-12
                  border-2 border-dashed rounded-xl transition-smooth
                  ${isDragging
                    ? "border-primary bg-primary/5 scale-105"
                    : "border-border"
                  }
                `}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <div className="w-24 h-24 rounded-full bg-gradient-primary flex items-center justify-center shadow-lg">
                  <Upload className="w-12 h-12 text-white" />
                </div>
                <div className="text-center max-w-md">
                  <h2 className="text-3xl font-bold mb-3">
                    Upload Your Dog's Photo
                  </h2>
                  <p className="text-muted-foreground text-lg mb-6">
                    Drag and drop your image here, or click the button below to
                    browse
                  </p>
                  <label htmlFor="file-upload">
                    <Button
                      size="lg"
                      type="button"
                      className="relative overflow-hidden group hover-lift"
                      onClick={() =>
                        document.getElementById("file-upload")?.click()
                      }
                    >
                      <Upload className="w-5 h-5 mr-2" />
                      Choose File
                    </Button>
                    <input
                      id="file-upload"
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleFileSelect}
                    />
                  </label>
                </div>
              </div>
            </Card>
          )}

          {/* Step 2: Select Analysis Type */}
          {currentStep === 2 && (
            <div className="space-y-6 animate-scale-in">
              <Card className="p-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-20 h-20 rounded-xl overflow-hidden border-2 border-primary shadow-lg">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold mb-2">
                      Select Analysis Type
                    </h2>
                    <p className="text-muted-foreground">
                      Choose how you'd like to analyze this image
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <button
                    onClick={() => setAnalysisType("basic")}
                    className={`
                      p-6 rounded-xl border-2 transition-smooth text-left hover-lift
                      ${analysisType === "basic"
                        ? "border-primary bg-primary/5 shadow-lg"
                        : "border-border hover:border-primary/50"
                      }
                    `}
                  >
                    <Search className="w-8 h-8 text-primary mb-3" />
                    <h3 className="font-semibold text-lg mb-2">Basic</h3>
                    <p className="text-sm text-muted-foreground">
                      Quick breed identification with confidence scores
                    </p>
                  </button>

                  <button
                    onClick={() => setAnalysisType("vlm")}
                    className={`
                      p-6 rounded-xl border-2 transition-smooth text-left hover-lift
                      ${analysisType === "vlm"
                        ? "border-primary bg-primary/5 shadow-lg"
                        : "border-border hover:border-primary/50"
                      }
                    `}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <Brain className="w-8 h-8 text-primary mb-3" />
                        <h3 className="font-semibold text-lg mb-2">VLM</h3>
                        <p className="text-sm text-muted-foreground">
                          Interactive Q&A about your dog using AI vision
                        </p>
                      </div>
                    </div>
                  </button>

                  <button
                    onClick={() => setAnalysisType("reasoning")}
                    className={`
                      p-6 rounded-xl border-2 transition-smooth text-left hover-lift
                      ${analysisType === "reasoning"
                        ? "border-primary bg-primary/5 shadow-lg"
                        : "border-border hover:border-primary/50"
                      }
                    `}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <Info className="w-8 h-8 text-primary mb-3" />
                        <h3 className="font-semibold text-lg mb-2">Reasoning</h3>
                        <p className="text-sm text-muted-foreground">
                          Deep visual reasoning and comparative analysis using AI
                        </p>
                      </div>
                    </div>
                  </button>
                </div>

                <div className="flex gap-3">
                  <Button variant="outline" onClick={resetWizard}>
                    Change Image
                  </Button>
                  <Button
                    className="flex-1"
                    onClick={() => setCurrentStep(3)}
                  >
                    Continue
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </Card>
            </div>
          )}

          {/* Step 3: Analyze */}
          {currentStep === 3 && (
            <div className="space-y-6 animate-scale-in">
              <Card className="p-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-20 h-20 rounded-xl overflow-hidden border-2 border-primary shadow-lg">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold mb-2">Ready to Analyze</h2>
                    <p className="text-muted-foreground">
                      Analysis Type:{" "}
                      <span className="font-semibold text-primary capitalize">
                        {analysisType}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="flex gap-3">
                  <Button variant="outline" onClick={() => setCurrentStep(2)}>
                    Back
                  </Button>
                  <Button
                    className="flex-1 bg-gradient-primary hover:opacity-90"
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-2" />
                        Analyze Image
                      </>
                    )}
                  </Button>
                </div>
              </Card>
            </div>
          )}

          {/* Step 4: Results */}
          {currentStep === 4 && results && (
            <div className="space-y-6 animate-fade-in">
              {/* Results Header */}
              <Card className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-20 h-20 rounded-xl overflow-hidden border-2 border-primary shadow-lg">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">
                        Analysis Complete
                      </h2>
                      <p className="text-muted-foreground">
                        {results.predictions && results.predictions.length > 0
                          ? `Top match: ${results.predictions[0].breed.replace(/_/g, " ")}`
                          : "Results ready"}
                      </p>
                    </div>
                  </div>
                  <Button variant="outline" onClick={resetWizard}>
                    New Analysis
                  </Button>
                </div>
              </Card>

              {/* Results Tabs */}
              <Card className="p-6">
                <Tabs defaultValue={getTabConfig().defaultTab}>
                  <TabsList className={`grid w-full mb-6 ${getTabConfig().visibleTabs.length === 1 ? 'grid-cols-1' : getTabConfig().visibleTabs.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
                    {getTabConfig().visibleTabs.includes("overview") && (
                      <TabsTrigger value="overview">Overview</TabsTrigger>
                    )}
                    {getTabConfig().visibleTabs.includes("details") && (
                      <TabsTrigger value="details">Details</TabsTrigger>
                    )}
                    {getTabConfig().visibleTabs.includes("queries") && (
                      <TabsTrigger value="queries">Q&A</TabsTrigger>
                    )}
                    {getTabConfig().visibleTabs.includes("analysis") && (
                      <TabsTrigger value="analysis">Visual Analysis</TabsTrigger>
                    )}
                  </TabsList>

                  <TabsContent value="overview" className="space-y-4">
                    <h3 className="text-xl font-semibold mb-4">
                      Top Predictions
                    </h3>
                    {results.predictions &&
                      results.predictions.map((pred, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-4 p-4 rounded-xl bg-gradient-soft border border-border hover-lift"
                        >
                          <div className="flex-shrink-0">
                            <div className="w-16 h-16 rounded-full bg-gradient-primary flex items-center justify-center text-white font-bold text-lg shadow-lg">
                              {Math.round(pred.confidence * 100)}%
                            </div>
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-lg capitalize mb-1">
                              {pred.breed.replace(/_/g, " ")}
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              {pred.info?.characteristics?.join(", ") ||
                                "No characteristics available"}
                            </p>
                          </div>
                        </div>
                      ))}
                  </TabsContent>

                  <TabsContent value="details" className="space-y-6">
                    {results.predictions && results.predictions[0] && (
                      <>
                        <div>
                          <h3 className="text-2xl font-bold capitalize mb-4">
                            {results.predictions[0].breed.replace(/_/g, " ")}
                          </h3>
                          {results.predictions[0].description && (
                            <p className="text-muted-foreground leading-relaxed p-4 bg-secondary/10 rounded-xl">
                              {results.predictions[0].description}
                            </p>
                          )}
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 rounded-xl bg-gradient-soft border border-border">
                            <h4 className="font-semibold text-primary mb-2">
                              Size
                            </h4>
                            <p>
                              {results.predictions[0].info?.size ||
                                "Not specified"}
                            </p>
                          </div>
                          <div className="p-4 rounded-xl bg-gradient-soft border border-border">
                            <h4 className="font-semibold text-primary mb-2">
                              Energy Level
                            </h4>
                            <p>
                              {results.predictions[0].info?.energy_level ||
                                "Not specified"}
                            </p>
                          </div>
                          <div className="p-4 rounded-xl bg-gradient-soft border border-border">
                            <h4 className="font-semibold text-primary mb-2">
                              Good with Children
                            </h4>
                            <p>
                              {results.predictions[0].info
                                ?.good_with_children || "Not specified"}
                            </p>
                          </div>
                          <div className="p-4 rounded-xl bg-gradient-soft border border-border">
                            <h4 className="font-semibold text-primary mb-2">
                              Trainability
                            </h4>
                            <p>
                              {results.predictions[0].info?.trainability ||
                                "Not specified"}
                            </p>
                          </div>
                        </div>

                        <div>
                          <h4 className="font-semibold mb-3">
                            Characteristics
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {results.predictions[0].info?.characteristics?.map(
                              (char, idx) => (
                                <span
                                  key={idx}
                                  className="px-4 py-2 bg-primary/10 text-primary rounded-full text-sm font-medium border border-primary/20"
                                >
                                  {char}
                                </span>
                              )
                            ) || "No characteristics specified"}
                          </div>
                        </div>
                      </>
                    )}
                  </TabsContent>

                  <TabsContent value="analysis" className="space-y-6">
                    {results.visual_reasoning && (
                      <div className="p-6 rounded-xl bg-gradient-soft border border-border">
                        <h4 className="font-semibold text-primary mb-3 flex items-center gap-2">
                          <Brain className="w-5 h-5" />
                          Visual Reasoning
                        </h4>
                        <p className="leading-relaxed">
                          {results.visual_reasoning}
                        </p>
                      </div>
                    )}

                    {results.comparative_reasoning && (
                      <div className="p-6 rounded-xl bg-gradient-soft border border-border">
                        <h4 className="font-semibold text-primary mb-3 flex items-center gap-2">
                          <Sparkles className="w-5 h-5" />
                          Comparative Analysis
                        </h4>
                        <p className="leading-relaxed">
                          {results.comparative_reasoning}
                        </p>
                      </div>
                    )}

                    {results.colors && (
                      <div className="p-6 rounded-xl bg-gradient-soft border border-border">
                        <h4 className="font-semibold text-primary mb-3">
                          Color Analysis
                        </h4>
                        <p className="leading-relaxed">{results.colors}</p>
                      </div>
                    )}

                    {results.key_visual_features &&
                      results.key_visual_features.length > 0 && (
                        <div>
                          <h4 className="font-semibold mb-3">
                            Key Visual Features
                          </h4>
                          <div className="grid grid-cols-2 gap-3">
                            {results.key_visual_features.map(
                              (feature, idx) => (
                                <div
                                  key={idx}
                                  className="p-4 rounded-xl bg-gradient-soft border border-border flex items-center gap-3"
                                >
                                  <div className="w-2 h-2 rounded-full bg-primary"></div>
                                  <span>{feature}</span>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}
                  </TabsContent>

                  <TabsContent value="queries" className="space-y-6">
                    {/* Query Input */}
                    <div className="space-y-4">
                      <h4 className="font-semibold">Ask About This Dog</h4>
                      <Textarea
                        placeholder="Ask a question about this dog or breeds in general..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="min-h-[60px]"
                      />
                      <div className="flex flex-wrap gap-2">
                        {suggestedQueries.slice(0, 4).map((q, i) => (
                          <Button
                            key={i}
                            variant="outline"
                            size="sm"
                            onClick={() => setQuery(q)}
                            className="text-xs hover:bg-primary/10"
                          >
                            {q}
                          </Button>
                        ))}
                      </div>
                      <Button
                        className="w-full"
                        onClick={submitSmartQuery}
                        disabled={!query || isLoading}
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Processing...
                          </>
                        ) : (
                          <>
                            <Send className="w-4 h-4 mr-2" />
                            Ask Question
                          </>
                        )}
                      </Button>
                    </div>

                    {/* Query Response */}
                    {isVisualLoading ? (
                      <div className="flex flex-col items-center justify-center py-12 space-y-4">
                        <Loader2 className="h-12 w-12 animate-spin text-primary" />
                        <p className="text-muted-foreground">
                          Processing your question...
                        </p>
                      </div>
                    ) : (
                      results?.queryResponse && (
                        <div className="space-y-4 p-6 rounded-xl bg-gradient-soft border border-border">
                          <div>
                            <h5 className="font-semibold mb-2">
                              Your Question
                            </h5>
                            <p className="text-muted-foreground">
                              {results.queryResponse.query}
                            </p>
                          </div>

                          {results.queryResponse.is_general_question && (
                            <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-semibold px-3 py-1 rounded-full">
                              General Dog Question
                            </span>
                          )}

                          {results.queryResponse.is_visual_question && (
                            <span className="inline-block bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 text-xs font-semibold px-3 py-1 rounded-full">
                              Visual Analysis Question
                            </span>
                          )}

                          <div>
                            <h5 className="font-semibold mb-2">Answer</h5>
                            <p className="leading-relaxed">
                              {results.queryResponse.response}
                            </p>
                          </div>
                        </div>
                      )
                    )}
                  </TabsContent>
                </Tabs>
              </Card>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
