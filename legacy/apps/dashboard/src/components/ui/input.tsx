import * as React from "react";
import { cn } from "../../lib";
export function Input({ className, ...props }: React.ComponentProps<"input">) {
	return <input className={cn("flex h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-2 focus-visible:outline-primary", className)} {...props} />;
}
