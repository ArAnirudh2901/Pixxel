

import React from 'react'
import FeatureCard from './FeatureCard';

const Features = () => {

    const features = [
        {
            icon: "✂️",
            title: "Smart Crop & Resize",
            description:
                "Interactive cropping with aspect ratio constraints and intelligent resizing that preserves image quality across any dimension.",
        },
        {
            icon: "🎨",
            title: "Color & Light Adjustment",
            description:
                "Professional-grade brightness, contrast, saturation controls with real-time preview and auto-enhance capabilities.",
        },
        {
            icon: "🤖",
            title: "AI Background Removal",
            description:
                "Remove or replace backgrounds instantly using advanced AI that detects complex edges and fine details with precision.",
        },
        {
            icon: "🔧",
            title: "AI Content Editor",
            description:
                "Edit images with natural language prompts. Remove objects, change elements, or add new content using generative AI.",
        },
        {
            icon: "📏",
            title: "Image Extender",
            description:
                "Expand your canvas in any direction with AI-powered generative fill that seamlessly blends new content with existing images.",
        },
        {
            icon: "⬆️",
            title: "AI Upscaler",
            description:
                "Enhance image resolution up to 4x using AI upscaling technology that preserves details and reduces artifacts.",
        },
    ];

    return (
        <section className='py-24 sm:py-28' id='features'>
            <div className='mx-auto max-w-7xl px-6'>
                <div className='mx-auto mb-18 max-w-4xl text-center'>
                    <h2 className='mb-6 text-5xl font-black tracking-tight text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 bg-clip-text sm:text-6xl'
                    >
                        Powerful AI Features
                    </h2>

                    <p className='mx-auto max-w-3xl text-xl leading-9 text-slate-300 sm:text-2xl'>
                        Everything you need to create, edit and enhance images with
                        professional-grade tools powered by cutting-edge AI.
                    </p>
                </div>

                <div className='grid gap-6 md:grid-cols-2 xl:grid-cols-3'>
                    {features.map((feature, index) => {
                        return <FeatureCard key={feature.title} {...feature} delay={index * 100} />
                    })}
                </div>
            </div>
        </section>
    )
}

export default Features
