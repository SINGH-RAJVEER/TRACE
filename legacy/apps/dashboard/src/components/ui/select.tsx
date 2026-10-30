import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib";

export const Select = SelectPrimitive.Root;
export const SelectValue = SelectPrimitive.Value;
export function SelectTrigger({ children, className, ...props }: React.ComponentProps<typeof SelectPrimitive.Trigger>) {
	return <SelectPrimitive.Trigger className={cn("flex h-10 items-center justify-between gap-3 rounded-md border border-input bg-background px-3 text-sm focus:outline-2 focus:outline-primary", className)} {...props}>{children}<SelectPrimitive.Icon><ChevronDown size={14} /></SelectPrimitive.Icon></SelectPrimitive.Trigger>;
}
export function SelectContent({ children, ...props }: React.ComponentProps<typeof SelectPrimitive.Content>) {
	return <SelectPrimitive.Portal><SelectPrimitive.Content position="popper" className="z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-y-auto rounded-md border bg-background p-1 shadow-md" {...props}><SelectPrimitive.Viewport>{children}</SelectPrimitive.Viewport></SelectPrimitive.Content></SelectPrimitive.Portal>;
}
export function SelectItem({ children, ...props }: React.ComponentProps<typeof SelectPrimitive.Item>) {
	return <SelectPrimitive.Item className="relative flex cursor-default items-center rounded-sm py-2 pr-3 pl-8 text-sm outline-none data-[highlighted]:bg-muted" {...props}><SelectPrimitive.ItemIndicator className="absolute left-2"><Check size={14} /></SelectPrimitive.ItemIndicator><SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText></SelectPrimitive.Item>;
}
