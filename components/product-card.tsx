"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Star, ShoppingCart, Zap } from "lucide-react"
import VTOModal from "./vto-modal"

interface Product {
  id: string
  name: string
  category: string
  price: number
  image: string
  description: string
  rating: number
  link?: string
}

export default function ProductCard({ product }: { product: Product }) {
  const [isVTOOpen, setIsVTOOpen] = useState(false)

  return (
    <>
      <Card className="overflow-hidden border-border bg-card hover:shadow-lg transition-shadow">
        <div className="relative aspect-square overflow-hidden bg-muted">
          <img
            src={product.image || "/placeholder.svg"}
            alt={product.name}
            className="h-full w-full object-cover hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute top-2 right-2 rounded-full bg-accent/90 px-2 py-1">
            <p className="text-xs font-semibold text-accent-foreground">{product.category}</p>
          </div>
        </div>
        <div className="p-3">
          <h3 className="font-semibold text-foreground line-clamp-2">{product.name}</h3>
          <p className="mt-1 text-xs text-muted-foreground line-clamp-2">{product.description}</p>
          <div className="mt-2 flex items-center gap-1">
            <Star className="h-3 w-3 fill-accent text-accent" />
            <span className="text-xs font-medium text-foreground">{product.rating}</span>
          </div>
          <div className="mt-3 flex items-center justify-between gap-2">
            <span className="text-lg font-bold text-primary">${product.price}</span>
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="outline"
                className="h-8 w-8 p-0 bg-transparent"
                onClick={() => setIsVTOOpen(true)}
                title="Thử đồ trực tiếp"
              >
                <Zap className="h-3 w-3" />
              </Button>
              {product.link ? (
                <a
                  href={product.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  title="Mở trang sản phẩm"
                >
                  <Button
                    size="sm"
                    className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90"
                  >
                    <ShoppingCart className="h-3 w-3" />
                  </Button>
                </a>
              ) : (
                <Button
                  size="sm"
                  className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90"
                  disabled
                  title="Không có link sản phẩm"
                >
                  <ShoppingCart className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </Card>

      <VTOModal product={product} isOpen={isVTOOpen} onClose={() => setIsVTOOpen(false)} />
    </>
  )
}
