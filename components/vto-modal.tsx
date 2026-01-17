"use client"

import { useState, useRef } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Upload, Loader2, X, Download } from "lucide-react"

interface VTOModalProps {
  product: {
    id: string
    name: string
    category: string
    image: string
    price: number
  }
  isOpen: boolean
  onClose: () => void
}

export default function VTOModal({ product, isOpen, onClose }: VTOModalProps) {
  const [poseImage, setPoseImage] = useState<string | null>(null)
  const [posePreview, setPosePreview] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file")
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must not exceed 10MB")
      return
    }

    setError(null)

    // Read file as base64
    const reader = new FileReader()
    reader.onload = (event) => {
      const base64 = event.target?.result as string
      setPoseImage(base64)
      setPosePreview(base64)
      setResultImage(null) // Clear previous result
    }
    reader.readAsDataURL(file)
  }

  const handleTryOn = async () => {
    if (!poseImage) {
      setError("Please upload your full-body image first")
      return
    }

    setIsProcessing(true)
    setError(null)
    setResultImage(null)

    try {
      // Call VTO API
      const baseUrl = "http://127.0.0.1:8000" // Backend URL
      const url = `${baseUrl}/vto`

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          person_image: poseImage,
          cloth_image: product.image, // Product image URL
          product_id: product.id,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: "Unknown error" }))
        throw new Error(errorData.message || `HTTP ${response.status}`)
      }

      const data = await response.json()

      if (data.status === "success" && data.result_image) {
        setResultImage(data.result_image)
      } else {
        throw new Error(data.message || "VTO processing failed")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred while processing the image")
      console.error("[VTO] Error:", err)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleRemovePose = () => {
    setPoseImage(null)
    setPosePreview(null)
    setResultImage(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleDownload = () => {
    if (!resultImage) return
    const link = document.createElement("a")
    link.href = resultImage
    link.download = `vto-${product.name.replace(/\s+/g, "-")}-${Date.now()}.png`
    link.click()
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl border-border bg-card max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-foreground">Virtual Try-On - {product.name}</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Product Info */}
          <div className="rounded-lg bg-muted/30 p-4">
            <div className="flex items-center gap-4">
              <img
                src={product.image}
                alt={product.name}
                className="h-20 w-20 rounded-lg object-cover border border-border"
              />
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">{product.name}</h3>
                <p className="text-sm text-muted-foreground">{product.category}</p>
                <p className="text-lg font-bold text-primary mt-1">${product.price}</p>
              </div>
            </div>
          </div>

          {/* Upload Pose Image Section */}
          <div>
              <label className="mb-3 block text-sm font-semibold text-foreground">
              Upload your full-body photo
            </label>
            {!posePreview ? (
              <div
                className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-sm text-foreground mb-2">
                  Click to choose an image or drag and drop here
                </p>
                <p className="text-xs text-muted-foreground">
                  Supported: JPG, PNG (up to 10MB)
                </p>
                <Input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="relative">
                <img
                  src={posePreview}
                  alt="Pose preview"
                  className="w-full h-auto rounded-lg border border-border max-h-96 object-contain"
                />
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2"
                  onClick={handleRemovePose}
                >
                  <X className="h-4 w-4" />
                </Button>
            </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-3">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {/* Try On Button */}
          <Button
            onClick={handleTryOn}
            disabled={!poseImage || isProcessing}
            className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90"
            size="lg"
          >
            {isProcessing ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              "Try on now"
            )}
          </Button>

          {/* Result Display */}
          {resultImage && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-foreground">Try-on result</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Original Pose */}
                {posePreview && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">Original photo</p>
                    <img
                      src={posePreview}
                      alt="Original pose"
                      className="w-full rounded-lg border border-border"
                    />
                  </div>
                )}
                {/* Product */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2">Product</p>
                  <img
                    src={product.image}
                    alt={product.name}
                    className="w-full rounded-lg border border-border"
                  />
                </div>
                {/* Result */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2">Result</p>
                  <img
                    src={resultImage}
                    alt="VTO Result"
                    className="w-full rounded-lg border-2 border-primary"
                  />
              </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-4 border-t border-border">
            <Button onClick={onClose} variant="outline" className="flex-1">
              Close
            </Button>
            {resultImage && (
              <Button
                onClick={handleDownload}
                variant="outline"
                className="flex-1"
              >
                <Download className="h-4 w-4 mr-2" />
                Tải ảnh kết quả
            </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
