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

  const getTabConfig = (): TabConfig => {
    switch (analysisType) {
      case "basic":
        return { defaultTab: "overview", visibleTabs: ["overview", "details"] };
      case "vlm":
        return { defaultTab: "overview", visibleTabs: ["overview", "queries"] };
      case "reasoning":
        return { defaultTab: "overview", visibleTabs: ["overview", "visual"] };
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

    try {
      // If no image is selected, it must be a general question
      if (!selectedImage) {
        await submitGeneralQuestion();
        return;
      }

      setIsLoading(true);
      setIsVisualLoading(true);
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("query", query);

      // Send to the VLM query endpoint - it will handle routing to the appropriate model
      const response = await fetch("/api/vision-language/query", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      // The backend now handles including visual data with responses
      // Set the results with all the data we received
      setResults((prev) =>
        prev
          ? {
              ...prev,
              queryResponse: data,
              // Update any visual data that came with the response
              ...(data.caption && { caption: data.caption }),
              ...(data.colors && { colors: data.colors }),
              ...(data.detailed_appearance && {
                detailed_appearance: data.detailed_appearance,
              }),
            }
          : data
      );

      // Clear the query input after submission
      setQuery("");
    } catch (error) {
      console.error("Error submitting query:", error);
      toast.error("Failed to process query. Please try again.");
    } finally {
      setIsLoading(false);
      setIsVisualLoading(false);
    }
  };

  const submitGeneralQuestion = async () => {
    if (!query) return;

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("query", query);

      const response = await fetch("/api/general-qa/query", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      if (!data.success && data.error) {
        toast.error(`Query failed: ${data.error}`);
        return;
      }

      // Create a queryResponse object even for general questions
      setResults((prev) =>
        prev
          ? {
              ...prev,
              queryResponse: {
                ...data,
                is_general_question: true,
                top_breed: prev.top_breed || "Unknown",
              },
            }
          : {
              success: true,
              queryResponse: {
                ...data,
                is_general_question: true,
                top_breed: "Unknown",
              },
            }
      );

      // Clear the query input after submission
      setQuery("");
    } catch (error) {
      console.error("Error submitting general question:", error);
      toast.error("Failed to process question. Please try again.");
    } finally {
      setIsLoading(false);
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

  return (
    <main className="h-screen overflow-auto bg-gradient-to-b from-background to-secondary/30">
      <div className="container mx-auto max-w-7xl flex flex-col min-h-screen">
        {/* Header */}
        <div className="flex items-center justify-between py-2 sticky top-0 bg-background/80 backdrop-blur-sm z-10">
          <Link href="/">
            <Button variant="ghost" className="gap-2 hover:bg-secondary/40">
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-primary">PawSense</h1>
            <Brain className="w-6 h-6 text-primary" />
            <PawPrint className="w-4 h-4 text-primary" />
          </div>
        </div>

        {/* Main Content - Split Layout */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 pb-4">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-5 flex flex-col">
            {/* Upload Card */}
            <Card
              className={`
              mb-2 p-4 border-2 transition-all duration-300 shadow-sm
          ${
            isDragging
              ? "border-primary border-dashed scale-105"
              : "border-border"
          }
              ${imagePreview ? "bg-muted/30 h-72 w-72 mx-auto " : ""}
        `}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              {!imagePreview ? (
                <div className="flex flex-col items-center gap-4 py-4 h-full justify-center">
                  <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center">
                    <Upload className="w-8 h-8 text-primary" />
                  </div>
                  <div className="text-center">
                    <h2 className="text-xl font-semibold mb-2">
                      Upload Your Dog's Photo
                    </h2>
                    <p className="text-muted-foreground mb-4">
                      Drag and drop your image here or click to browse
                    </p>
                    <label htmlFor="file-upload">
                      <Button
                        size="lg"
                        type="button"
                        className="relative overflow-hidden group"
                        onClick={() =>
                          document.getElementById("file-upload")?.click()
                        }
                      >
                        <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                        <Upload className="w-4 h-4 mr-2" />
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
              ) : (
                <div className="h-full flex flex-col">
                  <div className="relative aspect-square rounded-lg overflow-hidden border border-border shadow-sm mb-2">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex justify-between mt-auto">
                    <Button
                      variant="outline"
                      onClick={() => {
                        setImagePreview("");
                        setSelectedImage(null);
                        setResults(null);
                      }}
                    >
                      Replace
                    </Button>
                    <Button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="relative overflow-hidden"
                    >
                      {isAnalyzing ? (
                        <span className="flex items-center">
                          <svg
                            className="animate-spin -ml-1 mr-3 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Analyzing...
                        </span>
                      ) : (
                        "Analyze Image"
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </Card>

            {/* Analysis and Query Section - Condensed */}
            <Card className="mb-4 p-4 shadow-sm">
              <div className="space-y-2 flex flex-col">
                <div className="space-y-1 flex-shrink-0">
                  <h3 className="text-lg font-medium">Analysis Type</h3>
                  <div className="grid grid-cols-3 gap-2">
                    <Button
                      variant={analysisType === "basic" ? "default" : "outline"}
                      onClick={() => setAnalysisType("basic")}
                      className="flex items-center justify-center py-1 h-auto"
                      size="sm"
                    >
                      <Search className="w-4 h-4 mr-1" />
                      <span>Basic</span>
                    </Button>
                    <Button
                      variant={analysisType === "vlm" ? "default" : "outline"}
                      onClick={() => setAnalysisType("vlm")}
                      className="flex items-center justify-center py-1 h-auto"
                      size="sm"
                    >
                      <Brain className="w-4 h-4 mr-1" />
                      <span>VLM</span>
                    </Button>
                    <Button
                      variant={
                        analysisType === "reasoning" ? "default" : "outline"
                      }
                      onClick={() => setAnalysisType("reasoning")}
                      className="flex items-center justify-center py-1 h-auto"
                      size="sm"
                    >
                      <Info className="w-4 h-4 mr-1" />
                      <span>Reasoning</span>
                    </Button>
                  </div>
                </div>

                <h3 className="text-lg font-medium mt-2">Ask About This Dog</h3>
                <div className="flex-1 flex flex-col max-h-[200px]">
                  <Textarea
                    placeholder="Ask a question about this dog or breeds in general..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="min-h-[60px] flex-1 focus:border-primary focus:ring-primary max-h-[80px]"
                  />

                  <div className="flex flex-wrap gap-2 my-2">
                    {suggestedQueries.slice(0, 4).map((q, i) => (
                      <Button
                        key={i}
                        variant="outline"
                        size="sm"
                        onClick={() => setQuery(q)}
                        className="text-xs border border-secondary/50 hover:bg-secondary/30 hover:text-foreground"
                      >
                        {q}
                      </Button>
                    ))}
                  </div>

                  <div className="flex mt-auto">
                    <Button
                      className="w-full relative overflow-hidden group"
                      onClick={() =>
                        selectedImage
                          ? submitSmartQuery()
                          : submitGeneralQuestion()
                      }
                      disabled={!query}
                    >
                      <Send className="w-4 h-4 mr-2" />
                      Ask Question
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-7">
            <Card className="shadow-sm flex flex-col mb-4">
              {!results ? (
                <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
                  <div className="w-20 h-20 rounded-full bg-secondary/40 flex items-center justify-center mb-4">
                    <Dog className="w-10 h-10 text-secondary-foreground/60" />
                  </div>
                  <h2 className="text-2xl font-semibold mb-2">
                    Upload & Analyze
                  </h2>
                  <p className="text-muted-foreground max-w-md">
                    Upload a dog image and select an analysis type to see breed
                    identification, breed details, and more.
                  </p>
                </div>
              ) : (
                <Tabs
                  defaultValue={getTabConfig().defaultTab}
                  className="flex flex-col"
                >
                  <div className="px-4 pt-2 flex-shrink-0">
                    <TabsList className="grid grid-cols-3 w-full">
                      <TabsTrigger
                        value="overview"
                        className={
                          getTabConfig().visibleTabs.includes("overview")
                            ? ""
                            : "hidden"
                        }
                      >
                        Overview
                      </TabsTrigger>
                      <TabsTrigger
                        value="details"
                        className={
                          getTabConfig().visibleTabs.includes("details")
                            ? ""
                            : "hidden"
                        }
                      >
                        Details
                      </TabsTrigger>
                      <TabsTrigger
                        value="queries"
                        className={
                          getTabConfig().visibleTabs.includes("queries")
                            ? ""
                            : "hidden"
                        }
                      >
                        Queries
                      </TabsTrigger>
                      <TabsTrigger
                        value="visual"
                        className={
                          getTabConfig().visibleTabs.includes("visual")
                            ? ""
                            : "hidden"
                        }
                      >
                        Visual
                      </TabsTrigger>
                    </TabsList>
                  </div>

                  <div className="flex-1 p-4 overflow-auto">
                    <TabsContent
                      value="overview"
                      className="h-full m-0 space-y-3 overflow-auto"
                    >
                      <div className="space-y-3">
                        <h2 className="text-2xl font-bold">Top Predictions</h2>
                        {results.predictions &&
                          results.predictions.map((pred, idx) => (
                            <div
                              key={idx}
                              className="flex items-center space-x-4 p-4 bg-background rounded-lg shadow-sm hover:shadow-md transition-shadow"
                            >
                              <div className="w-16 text-center">
                                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary/10 text-primary font-medium">
                                  {Math.round(pred.confidence * 100)}%
                                </div>
                              </div>
                              <div className="flex-1">
                                <h3 className="font-semibold text-lg capitalize">
                                  {pred.breed.replace(/_/g, " ")}
                                </h3>
                                <p className="text-sm text-muted-foreground">
                                  {pred.info?.characteristics?.join(", ")}
                                </p>
                              </div>
                            </div>
                          ))}
                      </div>
                    </TabsContent>

                    <TabsContent
                      value="details"
                      className="h-full m-0 overflow-auto"
                    >
                      {results.predictions && results.predictions[0] && (
                        <div className="space-y-6">
                          <h2 className="text-2xl font-bold capitalize">
                            {results.predictions[0].breed.replace(/_/g, " ")}
                          </h2>

                          {results.predictions[0].description && (
                            <div className="p-4 bg-secondary/20 rounded-lg shadow-sm">
                              <p>{results.predictions[0].description}</p>
                            </div>
                          )}

                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2 p-3 bg-background rounded-lg shadow-sm">
                              <h3 className="font-medium text-primary">Size</h3>
                              <p>
                                {results.predictions[0].info?.size ||
                                  "Not specified"}
                              </p>
                            </div>
                            <div className="space-y-2 p-3 bg-background rounded-lg shadow-sm">
                              <h3 className="font-medium text-primary">
                                Energy Level
                              </h3>
                              <p>
                                {results.predictions[0].info?.energy_level ||
                                  "Not specified"}
                              </p>
                            </div>
                            <div className="space-y-2 p-3 bg-background rounded-lg shadow-sm">
                              <h3 className="font-medium text-primary">
                                Good with Children
                              </h3>
                              <p>
                                {results.predictions[0].info
                                  ?.good_with_children || "Not specified"}
                              </p>
                            </div>
                            <div className="space-y-2 p-3 bg-background rounded-lg shadow-sm">
                              <h3 className="font-medium text-primary">
                                Trainability
                              </h3>
                              <p>
                                {results.predictions[0].info?.trainability ||
                                  "Not specified"}
                              </p>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <h3 className="font-medium">Characteristics</h3>
                            <div className="flex flex-wrap gap-2">
                              {results.predictions[0].info?.characteristics?.map(
                                (char, idx) => (
                                  <span
                                    key={idx}
                                    className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm shadow-sm"
                                  >
                                    {char}
                                  </span>
                                )
                              ) || "No characteristics specified"}
                            </div>
                          </div>
                        </div>
                      )}
                    </TabsContent>

                    <TabsContent
                      value="visual"
                      className="h-full m-0 overflow-auto"
                    >
                      <div className="space-y-6">
                        {results.visual_reasoning && (
                          <div className="p-4 bg-secondary/20 rounded-lg shadow-sm">
                            <h3 className="font-medium mb-2 text-primary">
                              Visual Reasoning
                            </h3>
                            <p>{results.visual_reasoning}</p>
                          </div>
                        )}

                        {results.comparative_reasoning && (
                          <div className="p-4 bg-secondary/20 rounded-lg shadow-sm">
                            <h3 className="font-medium mb-2 text-primary">
                              Comparative Analysis
                            </h3>
                            <p>{results.comparative_reasoning}</p>
                          </div>
                        )}

                        {results.colors && (
                          <div className="p-4 bg-secondary/20 rounded-lg shadow-sm">
                            <h3 className="font-medium mb-2 text-primary">
                              Color Analysis
                            </h3>
                            <p>{results.colors}</p>
                          </div>
                        )}

                        {results.key_visual_features &&
                          results.key_visual_features.length > 0 && (
                            <div className="space-y-2">
                              <h3 className="font-medium text-primary">
                                Key Visual Features
                              </h3>
                              <div className="grid grid-cols-2 gap-2">
                                {results.key_visual_features.map(
                                  (feature, idx) => (
                                    <div
                                      key={idx}
                                      className="p-3 bg-background rounded-lg flex items-center gap-2 shadow-sm"
                                    >
                                      <div className="w-2 h-2 rounded-full bg-primary"></div>
                                      <span>{feature}</span>
                                    </div>
                                  )
                                )}
                              </div>
                            </div>
                          )}
                      </div>
                    </TabsContent>

                    <TabsContent
                      value="queries"
                      className="h-full m-0 overflow-auto"
                    >
                      {isVisualLoading ? (
                        <div className="flex flex-col items-center justify-center py-10 space-y-4">
                          <Loader2 className="h-10 w-10 animate-spin text-primary" />
                          <p className="text-center text-sm text-muted-foreground">
                            Processing your question... Please wait.
                          </p>
                        </div>
                      ) : (
                        <>
                          {results?.queryResponse ? (
                            <div className="space-y-4">
                              <div>
                                <h3 className="font-semibold mb-1">
                                  Your Question
                                </h3>
                                <p className="text-muted-foreground">
                                  {results.queryResponse.query}
                                </p>
                              </div>

                              {results.queryResponse.is_general_question && (
                                <div className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-semibold px-2 py-1 rounded">
                                  General Dog Question
                                </div>
                              )}

                              {results.queryResponse.is_visual_question && (
                                <div className="inline-block bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 text-xs font-semibold px-2 py-1 rounded">
                                  Visual Analysis Question
                                </div>
                              )}

                              <div>
                                <h3 className="font-semibold mb-1">Answer</h3>
                                <p>{results.queryResponse.response}</p>
                              </div>
                            </div>
                          ) : (
                            <p className="text-center text-sm text-muted-foreground py-4">
                              Ask a question about the dog in the image
                            </p>
                          )}
                        </>
                      )}
                    </TabsContent>
                  </div>
                </Tabs>
              )}
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
