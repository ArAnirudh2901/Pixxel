import React from 'react'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Crown, Zap } from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';
import { PricingTable } from '@clerk/nextjs';
import { pricingTableAppearance } from '@/lib/pricing-table-appearance';
import { Button } from './ui/button';

const UpgradeModel = ({ isOpen, onClose, restrictedTool, reason }) => {

    const getToolName = (toolId) => {
        const toolNames = {
            background: "AI Background Tools",
            ai_extender: "AI Image Extender",
            ai_edit: "AI Editor",
            projects: "Unlimited Projects",
        }
        return toolNames[toolId] || "Premium Feature"
    }

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-5xl bg-slate-950/95 border-white/10 shadow-[0_32px_120px_rgba(2,6,23,0.78)] backdrop-blur-xl max-h-[90vh] overflow-y-auto p-6 sm:p-8">
                <DialogHeader>
                    <div className='flex items-center gap-3'>
                        <Crown className="h-6 w-6 text-yellow-500" />
                        <DialogTitle className="text-2xl font-bold text-white">
                            Upgrade to Pro
                        </DialogTitle>
                    </div>
                </DialogHeader>
                <div className='space-y-6'>
                    {restrictedTool &&
                        <Alert className="bg-amber-500/10 border-amber-500/20">
                            <Zap className='h-5 w-5 text-amber-400' />
                            <AlertDescription className="text-amber-300/80">
                                <div className='font-semibold text-amber-400 mb-1'>
                                    {getToolName(restrictedTool)} - Pro Feature
                                </div>
                                {reason ||
                                    `Upgrade now to unlock ${getToolName(restrictedTool)}`
                                }
                            </AlertDescription>
                        </Alert>
                    }
                    <div className='pricing-table-shell pricing-table-shell--upgrade w-full'>
                        <PricingTable
                            appearance={pricingTableAppearance}
                            checkoutProps={{
                                appearance: {
                                    elements: {
                                        drawerRoot: {
                                            zIndex: 200000,
                                        },
                                    },
                                },
                            }}
                        />
                    </div>
                </div>

                <DialogFooter className="justify-center">
                    <Button
                        variant='ghost'
                        onClick={onClose}
                        className="text-white/70 hover:text-white"
                    >
                        Maybe Later

                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};

export default UpgradeModel
