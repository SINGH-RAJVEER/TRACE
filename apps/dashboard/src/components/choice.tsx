import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

export function Choice({ value, onChange, options, label, id }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string; id?: string }) {
	return <Select value={value} onValueChange={onChange}>
		<SelectTrigger id={id} aria-label={label}><SelectValue /></SelectTrigger>
		<SelectContent>{options.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
	</Select>;
}
