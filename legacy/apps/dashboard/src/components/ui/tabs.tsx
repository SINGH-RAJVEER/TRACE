import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "../../lib";
export const Tabs = TabsPrimitive.Root;
export function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
	return <TabsPrimitive.List className={cn("inline-flex max-w-full items-center gap-1 overflow-x-auto rounded-md bg-muted p-1", className)} {...props} />;
}
export function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
	return <TabsPrimitive.Trigger className={cn("whitespace-nowrap rounded-sm px-4 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-primary data-[state=active]:bg-background data-[state=active]:shadow-sm", className)} {...props} />;
}
export function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
	return <TabsPrimitive.Content className={cn("mt-6 focus-visible:outline-2 focus-visible:outline-primary", className)} {...props} />;
}
