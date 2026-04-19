"use client"

import { Toaster as Sonner } from "sonner";
import { CheckCircleIcon, InfoIcon, WarningIcon, XCircleIcon, SpinnerIcon } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"

const defaultIcons = {
  success: (
    <CheckCircleIcon className="size-[18px]" weight="fill" />
  ),
  info: (
    <InfoIcon className="size-[18px]" weight="fill" />
  ),
  warning: (
    <WarningIcon className="size-[18px]" weight="fill" />
  ),
  error: (
    <XCircleIcon className="size-[18px]" weight="fill" />
  ),
  loading: (
    <SpinnerIcon className="size-[18px] animate-spin" weight="bold" />
  ),
}

const defaultToastClassNames = {
  toast: cn(
    "cn-toast",
    "relative isolate flex items-center gap-3 overflow-hidden rounded-[24px] border border-white/12",
    "bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(15,23,42,0.82))] px-4 py-4 text-white",
    "shadow-[0_24px_70px_rgba(2,6,23,0.42)] backdrop-blur-2xl supports-[backdrop-filter]:bg-slate-950/72",
    "transition-[border-color,box-shadow,background] duration-300 ease-out"
  ),
  content: "cn-toast-content min-w-0 flex-1",
  title: "cn-toast-title text-[15px] font-semibold tracking-[-0.02em] text-white",
  description: "cn-toast-description mt-1 text-[13px] leading-5 text-slate-300",
  icon: cn(
    "cn-toast-icon flex size-10 shrink-0 items-center justify-center rounded-[16px] border border-white/10",
    "bg-white/[0.07] text-cyan-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]",
    "transition-[transform,border-color,background-color,box-shadow] duration-300 ease-out"
  ),
  actionButton: cn(
    "cn-toast-action inline-flex h-10 shrink-0 items-center justify-center rounded-full border border-white/14",
    "bg-white/[0.08] px-4 text-sm font-semibold tracking-[-0.01em] text-white",
    "shadow-[inset_0_1px_0_rgba(255,255,255,0.1)] transition-all duration-200 ease-out",
    "hover:border-white/24 hover:bg-white/[0.14] hover:shadow-[0_12px_28px_rgba(2,6,23,0.24),inset_0_1px_0_rgba(255,255,255,0.16)]",
    "active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/25"
  ),
  cancelButton: cn(
    "cn-toast-cancel inline-flex h-10 shrink-0 items-center justify-center rounded-full border border-white/10",
    "bg-white/[0.04] px-4 text-sm font-medium tracking-[-0.01em] text-slate-200",
    "shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] transition-all duration-200 ease-out",
    "hover:border-white/18 hover:bg-white/[0.08] hover:text-white",
    "active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/15"
  ),
  closeButton: cn(
    "cn-toast-close absolute right-3 top-3 inline-flex size-7 items-center justify-center rounded-full border border-white/10",
    "bg-slate-950/70 text-slate-200 backdrop-blur-xl transition-all duration-200",
    "hover:border-white/18 hover:bg-slate-900/90 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/15"
  ),
  success: "cn-toast-success",
  error: "cn-toast-error",
  info: "cn-toast-info",
  warning: "cn-toast-warning",
  loading: "cn-toast-loading",
  default: "cn-toast-default",
}

const Toaster = ({
  className,
  icons,
  style,
  toastOptions,
  ...props
}) => {
  const mergedIcons = {
    ...defaultIcons,
    ...icons,
  }

  const mergedToastOptions = {
    duration: 5000,
    closeButton: false,
    unstyled: true,
    ...toastOptions,
    classNames: {
      ...defaultToastClassNames,
      ...(toastOptions?.classNames ?? {}),
    },
  }

  return (
    <Sonner
      theme="dark"
      position="top-center"
      expand
      visibleToasts={4}
      closeButton={false}
      offset={{ top: 92, right: 24, bottom: 24, left: 24 }}
      mobileOffset={{ top: 82, right: 16, bottom: 16, left: 16 }}
      className={cn("toaster", className)}
      icons={mergedIcons}
      style={
        {
          "--width": "min(420px, calc(100vw - 1.5rem))",
          "--normal-bg": "rgba(15,23,42,0.86)",
          "--normal-text": "rgba(248,250,252,0.98)",
          "--normal-border": "rgba(255,255,255,0.12)",
          "--border-radius": "24px",
          ...style,
        }
      }
      toastOptions={mergedToastOptions}
      {...props} />
  );
}

export { Toaster }
