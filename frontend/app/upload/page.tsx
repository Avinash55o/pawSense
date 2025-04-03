"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Upload, Brain, PawPrint, ArrowLeft } from "lucide-react";
import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import Link from "next/link";

export default function UploadPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/analyzer");
  }, [router]);

  return null;
}