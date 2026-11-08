import * as React from "react";
import { cn } from "../../lib";

export function Table({ className, ...props }: React.ComponentProps<"table">) {
	return <div className="w-full overflow-x-auto"><table className={cn("w-full caption-bottom border-collapse text-sm", className)} {...props} /></div>;
}

export function TableHeader({ className, ...props }: React.ComponentProps<"thead">) {
	return <thead className={cn("[&_tr]:border-b [&_tr]:border-rule", className)} {...props} />;
}

export function TableBody({ className, ...props }: React.ComponentProps<"tbody">) {
	return <tbody className={cn("[&_tr:last-child]:border-0", className)} {...props} />;
}

export function TableRow({ className, ...props }: React.ComponentProps<"tr">) {
	return <tr className={cn("border-b border-rule/70 hover:bg-wash/60", className)} {...props} />;
}

export function TableHead({ className, ...props }: React.ComponentProps<"th">) {
	return <th className={cn("h-9 px-2 text-left align-bottom text-xs font-medium whitespace-nowrap text-muted", className)} {...props} />;
}

export function TableCell({ className, ...props }: React.ComponentProps<"td">) {
	return <td className={cn("px-2 py-1.5 align-top", className)} {...props} />;
}

export function TableCaption({ className, ...props }: React.ComponentProps<"caption">) {
	return <caption className={cn("mt-2 text-left text-xs text-muted", className)} {...props} />;
}
