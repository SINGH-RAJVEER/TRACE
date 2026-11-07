import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib";

const badgeVariants = cva("inline-flex items-center gap-1 rounded-sm border px-1.5 py-px text-xs font-medium whitespace-nowrap", {
	variants: {
		variant: {
			outline: "border-rule text-muted",
			pass: "border-pass/40 text-pass",
			fail: "border-fail/50 text-fail",
			caution: "border-caution/50 text-caution",
			info: "border-accent/40 text-accent",
		},
	},
	defaultVariants: { variant: "outline" },
});

export function Badge({ className, variant, ...props }: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
	return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
