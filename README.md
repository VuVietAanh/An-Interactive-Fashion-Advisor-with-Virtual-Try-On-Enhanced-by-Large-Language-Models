# Multimodal LLM-Powered Fashion Recommender (CRS + Virtual Try-On)
## Project Summary

Built an end-to-end Conversational Recommender System (CRS) integrated with a diffusion-based Virtual Try-On (VTO) pipeline, enabling users to:

Search fashion products via natural language

Refine results through multi-turn dialogue

Virtually try garments on real human images

The system combines LLM fine-tuning, multimodal retrieval, and diffusion-based image generation in a unified architecture.

## Key Contributions
### Conversational Recommender System

🔹 Designed a multimodal semantic retrieval engine using:

> Sentence embeddings (SBERT)

> Image-text alignment (CLIP)

> FAISS for scalable similarity search

🔹 Fine-tuned Phi-2 with LoRA to:

> Support multi-turn dialogue

> Maintain conversation context

> Perform constraint-aware filtering (color, type, style, price)

→ Result: Natural, constraint-aware product search instead of keyword matching.

### Virtual Try-On Pipeline (Optimized Lightweight Architecture)

Built a cost-efficient diffusion-based try-on system with semantic guidance.

🔹 Semantic Control Layer

Used CLIP to classify garment regions (upper/lower body)

Used BLIP to generate garment captions for semantic conditioning

→ Improves inpainting consistency and reduces ambiguity.

🔹 Advanced Mask Optimization

Replaced CNN segmentation with SegFormer (Transformer-based) for precise garment boundary extraction

Applied Dilation Masking (20×20 kernel) to handle size mismatch and improve loose-fit realism

→ Higher boundary quality and better garment-body adaptation.

🔹 Diffusion Generation

Integrated CatVTON (lightweight inpainting diffusion model)

Tuned inference parameters for balance between realism and compute cost

### Hybrid Fallback Strategy

Default: Lightweight in-house VTO pipeline (cost-efficient, controllable)

Fallback: Advanced APIs (IDM-VTON / OOTDiffusion) for complex poses or occlusions

→ Practical engineering trade-off between cost, latency, and quality.
