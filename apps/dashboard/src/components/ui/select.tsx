import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib";

export const Select = SelectPrimitive.Root;
export const SelectValue = SelectPrimitive.Value;

export function SelectTrigger({ children, className, ...props }: React.ComponentProps<typeof SelectPrimitive.Trigger>) {
	return <SelectPrimitive.Trigger className={cn("flex h-10 w-full items-center justify-between gap-3 rounded-md border border-rule bg-paper px-3 text-left text-sm focus-visible:outline-2 focus-visible:outline-accent", className)} {...props}>{children}<SelectPrimitive.Icon><ChevronDown size={14} /></SelectPrimitive.Icon></SelectPrimitive.Trigger>;
}

export function SelectContent({ children, className, ...props }: React.ComponentProps<typeof SelectPrimitive.Content>) {
	return <SelectPrimitive.Portal><SelectPrimitive.Content position="popper" sideOffset={4} className={cn("z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-y-auto rounded-md border border-rule bg-paper p-1 text-ink shadow-lg", className)} {...props}><SelectPrimitive.Viewport>{children}</SelectPrimitive.Viewport></SelectPrimitive.Content></SelectPrimitive.Portal>;
}

export function SelectItem({ children, className, ...props }: React.ComponentProps<typeof SelectPrimitive.Item>) {
	return <SelectPrimitive.Item className={cn("relative flex cursor-default items-center rounded-sm py-2 pr-3 pl-8 text-sm outline-none data-[highlighted]:bg-wash", className)} {...props}><SelectPrimitive.ItemIndicator className="absolute left-2"><Check size={14} /></SelectPrimitive.ItemIndicator><SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText></SelectPrimitive.Item>;
}
