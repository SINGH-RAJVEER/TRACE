"""Project-authored prompt families generated from templates with computed answer keys.

Every family is deterministic for a given seed. Each generated prompt carries the reference
that review uses to accept or reject the teacher's answer, so these rows are verifiable.
"""

import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

GSM8K_SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."
HUMANEVAL_PREFIX = (
	"Complete this Python function. Return only the full Python code, "
	"including the function signature.\n\n"
)
MBPP_PREFIX = "Write Python code for this task. Return only Python code.\n\n"


@dataclass(frozen=True)
class Generated:
	"""One generated prompt. `reference` holds ReferenceRecord fields except example_id."""

	key: str
	group: str
	task: str
	user: str
	reference: dict[str, Any]
	system: str | None = None


# Python function synthesis. Expected outputs come from the reference implementation at
# build time, so every test is correct by construction.


@dataclass(frozen=True)
class FunctionSpec:
	name: str
	signature: str
	description: str
	implementation: Callable[..., Any]
	cases: list[tuple[Any, ...]] = field(default_factory=list)


def _run_length_encode(text: str) -> str:
	parts: list[str] = []
	for char in text:
		if parts and parts[-1][0] == char:
			parts[-1] = char + str(int(parts[-1][1:]) + 1)
		else:
			parts.append(char + "1")
	return "".join(parts)


def _run_length_decode(encoded: str) -> str:
	return "".join(char * int(count) for char, count in re.findall(r"(\D)(\d+)", encoded))


def _spiral(matrix: list[list[int]]) -> list[int]:
	result: list[int] = []
	rows = [list(row) for row in matrix]
	while rows:
		result.extend(rows.pop(0))
		rows = [list(row) for row in zip(*rows, strict=True)][::-1]
	return result


def _merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
	merged: list[list[int]] = []
	for start, end in sorted(intervals):
		if merged and start <= merged[-1][1]:
			merged[-1][1] = max(merged[-1][1], end)
		else:
			merged.append([start, end])
	return merged


def _word_frequencies(text: str) -> dict[str, int]:
	return dict(sorted(Counter(re.findall(r"[a-z]+", text.lower())).items()))


def _caesar(text: str, shift: int) -> str:
	def move(char: str) -> str:
		if char.isascii() and char.isalpha():
			base = ord("A") if char.isupper() else ord("a")
			return chr((ord(char) - base + shift) % 26 + base)
		return char

	return "".join(move(char) for char in text)


def _count_islands(grid: list[str]) -> int:
	seen: set[tuple[int, int]] = set()
	count = 0
	for row_index, row in enumerate(grid):
		for column_index, cell in enumerate(row):
			if cell != "#" or (row_index, column_index) in seen:
				continue
			count += 1
			stack = [(row_index, column_index)]
			while stack:
				r, c = stack.pop()
				if (r, c) in seen or not (0 <= r < len(grid) and 0 <= c < len(grid[r])):
					continue
				if grid[r][c] != "#":
					continue
				seen.add((r, c))
				stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
	return count


def _balanced(text: str) -> bool:
	pairs = {")": "(", "]": "[", "}": "{"}
	stack: list[str] = []
	for char in text:
		if char in "([{":
			stack.append(char)
		elif char in pairs and (not stack or stack.pop() != pairs[char]):
			return False
	return not stack


def _roman_to_int(numeral: str) -> int:
	values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
	total = 0
	for index, char in enumerate(numeral):
		value = values[char]
		if index + 1 < len(numeral) and values[numeral[index + 1]] > value:
			total -= value
		else:
			total += value
	return total


def _flatten(nested: list[Any]) -> list[Any]:
	result: list[Any] = []
	for item in nested:
		if isinstance(item, list):
			result.extend(_flatten(item))
		else:
			result.append(item)
	return result


def _group_anagrams(words: list[str]) -> list[list[str]]:
	groups: dict[str, list[str]] = {}
	for word in words:
		groups.setdefault("".join(sorted(word)), []).append(word)
	return sorted(sorted(group) for group in groups.values())


def _top_k(values: list[int], k: int) -> list[int]:
	counts = Counter(values)
	return [value for value, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k]]


def _insert_position(values: list[int], target: int) -> int:
	low, high = 0, len(values)
	while low < high:
		middle = (low + high) // 2
		if values[middle] < target:
			low = middle + 1
		else:
			high = middle
	return low


def _isbn10(code: str) -> bool:
	digits = code.replace("-", "")
	if not re.fullmatch(r"\d{9}[\dX]", digits):
		return False
	total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(digits))
	return total % 11 == 0


def _luhn(number: str) -> bool:
	digits = [int(c) for c in number if c.isdigit()]
	if len(digits) < 2 or any(not (c.isdigit() or c == " ") for c in number):
		return False
	total = 0
	for index, digit in enumerate(reversed(digits)):
		if index % 2:
			digit *= 2
			if digit > 9:
				digit -= 9
		total += digit
	return total % 10 == 0


def _format_duration(seconds: int) -> str:
	hours, rest = divmod(seconds, 3600)
	minutes, secs = divmod(rest, 60)
	parts = [f"{v}{u}" for v, u in ((hours, "h"), (minutes, "m"), (secs, "s")) if v]
	return " ".join(parts) or "0s"


def _parse_pairs(text: str) -> dict[str, str]:
	result: dict[str, str] = {}
	for segment in text.split(";"):
		if "=" in segment:
			key, value = segment.split("=", 1)
			result[key.strip()] = value.strip()
	return result


def _common_prefix(words: list[str]) -> str:
	if not words:
		return ""
	prefix = words[0]
	for word in words[1:]:
		while not word.startswith(prefix):
			prefix = prefix[:-1]
	return prefix


def _pascal_row(n: int) -> list[int]:
	return [math.comb(n, k) for k in range(n + 1)]


def _collatz(n: int) -> int:
	steps = 0
	while n != 1:
		n = n // 2 if n % 2 == 0 else 3 * n + 1
		steps += 1
	return steps


def _digital_root(n: int) -> int:
	while n >= 10:
		n = sum(int(d) for d in str(n))
	return n


def _mask_email(email: str) -> str:
	local, domain = email.split("@", 1)
	return f"{local[0]}***@{domain}"


def _title_case(text: str, minor: list[str]) -> str:
	words = text.lower().split()
	return " ".join(
		word if index and word in minor else word.capitalize() for index, word in enumerate(words)
	)


def _duplicates(values: list[int]) -> list[int]:
	return sorted(value for value, count in Counter(values).items() if count > 1)


def _interleave(first: list[Any], second: list[Any]) -> list[Any]:
	result: list[Any] = []
	for index in range(max(len(first), len(second))):
		if index < len(first):
			result.append(first[index])
		if index < len(second):
			result.append(second[index])
	return result


def _compress_ranges(values: list[int]) -> list[str]:
	result: list[str] = []
	index = 0
	while index < len(values):
		end = index
		while end + 1 < len(values) and values[end + 1] == values[end] + 1:
			end += 1
		result.append(str(values[index]) if end == index else f"{values[index]}-{values[end]}")
		index = end + 1
	return result


def _expand_ranges(spec: str) -> list[int]:
	result: list[int] = []
	for part in spec.split(","):
		part = part.strip()
		if not part:
			continue
		if "-" in part[1:]:
			start, end = part.split("-", 1)
			result.extend(range(int(start), int(end) + 1))
		else:
			result.append(int(part))
	return result


def _matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
	return [
		[sum(x * y for x, y in zip(row, col, strict=True)) for col in zip(*b, strict=True)]
		for row in a
	]


def _gcd_list(values: list[int]) -> int:
	return math.gcd(*values)


def _is_anagram(first: str, second: str) -> bool:
	def key(text: str) -> list[str]:
		return sorted(text.replace(" ", "").lower())

	return key(first) == key(second)


def _remove_adjacent(text: str) -> str:
	return "".join(char for index, char in enumerate(text) if index == 0 or text[index - 1] != char)


def _days_between(first: str, second: str) -> int:
	return abs((date.fromisoformat(second) - date.fromisoformat(first)).days)


def _weighted_average(values: list[float], weights: list[float]) -> float:
	return round(sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights), 3)


def _leaderboard(scores: dict[str, int]) -> list[str]:
	return [name for name, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


def _mode(values: list[int]) -> int:
	counts = Counter(values)
	best = max(counts.values())
	return min(value for value, count in counts.items() if count == best)


def _parse_csv_line(line: str) -> list[str]:
	fields: list[str] = []
	current = ""
	quoted = False
	for char in line:
		if char == '"':
			quoted = not quoted
		elif char == "," and not quoted:
			fields.append(current)
			current = ""
		else:
			current += char
	fields.append(current)
	return fields


def _max_depth(text: str) -> int:
	depth = best = 0
	for char in text:
		if char == "(":
			depth += 1
			best = max(best, depth)
		elif char == ")":
			depth -= 1
	return best


def _rotate(matrix: list[list[int]]) -> list[list[int]]:
	return [list(row) for row in zip(*matrix[::-1], strict=True)]


def _moving_average(values: list[float], window: int) -> list[float]:
	return [round(sum(values[i : i + window]) / window, 2) for i in range(len(values) - window + 1)]


def _second_largest(values: list[int]) -> int | None:
	distinct = sorted(set(values), reverse=True)
	return distinct[1] if len(distinct) > 1 else None


def _running_total(values: list[int]) -> list[int]:
	total = 0
	result: list[int] = []
	for value in values:
		total += value
		result.append(total)
	return result


def _histogram(counts: dict[str, int]) -> str:
	return "\n".join(f"{key}: {'*' * counts[key]}" for key in sorted(counts))


def _kth_smallest(values: list[int], k: int) -> int:
	return sorted(values)[k - 1]


def _snake_to_camel(name: str) -> str:
	head, *rest = name.split("_")
	return head + "".join(part.capitalize() for part in rest)


def _camel_to_snake(name: str) -> str:
	return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


FUNCTION_SPECS = [
	FunctionSpec(
		"run_length_encode",
		"(text: str) -> str",
		"Compress a string with run-length encoding, writing each character followed by the "
		'length of its run. For example, "aaabcc" becomes "a3b1c2". An empty string stays empty.',
		_run_length_encode,
		[("aaabcc",), ("",), ("abc",), ("zzzzzzzzzzzz",), ("aabbaa",)],
	),
	FunctionSpec(
		"run_length_decode",
		"(encoded: str) -> str",
		"Expand a run-length encoded string in which each character is followed by a decimal "
		'count, which may have several digits. For example, "a3b1c2" becomes "aaabcc".',
		_run_length_decode,
		[("a3b1c2",), ("",), ("x12",), ("m1n2o3",)],
	),
	FunctionSpec(
		"chunk_list",
		"(items: list, size: int) -> list[list]",
		"Split a list into consecutive chunks of the given size. The last chunk may be shorter.",
		lambda items, size: [items[i : i + size] for i in range(0, len(items), size)],
		[([1, 2, 3, 4, 5], 2), ([], 3), (["a", "b", "c"], 3), ([1, 2, 3], 5)],
	),
	FunctionSpec(
		"rotate_clockwise",
		"(matrix: list[list[int]]) -> list[list[int]]",
		"Return a new matrix equal to the input rotated 90 degrees clockwise. The input may be "
		"rectangular.",
		_rotate,
		[([[1, 2], [3, 4]],), ([[1, 2, 3]],), ([[1, 2, 3], [4, 5, 6]],)],
	),
	FunctionSpec(
		"spiral_order",
		"(matrix: list[list[int]]) -> list[int]",
		"Return the elements of a rectangular matrix in clockwise spiral order, starting at the "
		"top-left corner.",
		_spiral,
		[([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), ([[1, 2], [3, 4], [5, 6]],), ([[7]],)],
	),
	FunctionSpec(
		"merge_intervals",
		"(intervals: list[list[int]]) -> list[list[int]]",
		"Merge overlapping closed intervals given as [start, end] pairs and return them sorted "
		"by start. Intervals that touch, such as [1, 3] and [3, 5], also merge.",
		_merge_intervals,
		[([[1, 3], [2, 6], [8, 10]],), ([[5, 7], [1, 2], [2, 4]],), ([],), ([[1, 10], [2, 3]],)],
	),
	FunctionSpec(
		"word_frequencies",
		"(text: str) -> dict[str, int]",
		"Count how often each word occurs in the text. Words are maximal runs of ASCII letters, "
		"compared case-insensitively and returned in lowercase. Return the dictionary with keys "
		"in alphabetical order.",
		_word_frequencies,
		[("The cat and the hat.",), ("",), ("Go, go, GO!",), ("one-two two",)],
	),
	FunctionSpec(
		"caesar_shift",
		"(text: str, shift: int) -> str",
		"Shift every ASCII letter forward in the alphabet by `shift` positions, wrapping around "
		"and preserving case. Leave every other character unchanged. Negative shifts move "
		"backward.",
		_caesar,
		[("Hello, World!", 3), ("xyz", 2), ("abc", -1), ("Zebra 42", 27)],
	),
	FunctionSpec(
		"normalize_whitespace",
		"(text: str) -> str",
		"Replace every run of whitespace characters with a single space and remove leading and "
		"trailing whitespace.",
		lambda text: " ".join(text.split()),
		[("  hello   world \n",), ("\t\ta\tb  ",), ("",), ("single",)],
	),
	FunctionSpec(
		"second_largest",
		"(values: list[int]) -> int | None",
		"Return the second largest distinct value in the list, or None when fewer than two "
		"distinct values exist.",
		_second_largest,
		[([4, 1, 4, 3],), ([7, 7],), ([],), ([-2, -5, -1],)],
	),
	FunctionSpec(
		"running_total",
		"(values: list[int]) -> list[int]",
		"Return the cumulative sums of the list, so element i is the sum of values[0..i].",
		_running_total,
		[([1, 2, 3, 4],), ([],), ([5, -5, 5],)],
	),
	FunctionSpec(
		"moving_average",
		"(values: list[float], window: int) -> list[float]",
		"Return the average of every contiguous window of the given size, rounded to 2 decimal "
		"places. Return an empty list when the window is longer than the input.",
		_moving_average,
		[([1, 2, 3, 4, 5], 2), ([10, 20, 30], 3), ([1, 2], 3), ([1, 1, 2], 2)],
	),
	FunctionSpec(
		"count_islands",
		"(grid: list[str]) -> int",
		'Count the islands in a grid of "#" (land) and "." (water). Land cells belong to the '
		"same island when they touch horizontally or vertically. Rows may differ in length.",
		_count_islands,
		[(["##..", "#...", "..##"],), (["....", "...."],), (["#.#", ".#.", "#.#"],), ([],)],
	),
	FunctionSpec(
		"is_balanced",
		"(text: str) -> bool",
		"Return True when every bracket in the text, of the kinds (), [], and {}, is closed by "
		"the matching kind in the correct order. Ignore all other characters.",
		_balanced,
		[("a(b[c]{d})",), ("(]",), ("",), ("((",), ("{[()()]}x",)],
	),
	FunctionSpec(
		"roman_to_int",
		"(numeral: str) -> int",
		"Convert an uppercase Roman numeral using the standard subtractive notation to an integer.",
		_roman_to_int,
		[("III",), ("XLII",), ("MCMXCIV",), ("CDXLIV",)],
	),
	FunctionSpec(
		"flatten",
		"(nested: list) -> list",
		"Flatten arbitrarily nested lists into one list, keeping the left-to-right order of the "
		"non-list elements.",
		_flatten,
		[([1, [2, [3, [4]]], 5],), ([],), ([[[]], "a"],), ([[1, 2], [3]],)],
	),
	FunctionSpec(
		"group_anagrams",
		"(words: list[str]) -> list[list[str]]",
		"Group words that are anagrams of each other. Sort the words inside each group, then "
		"sort the list of groups.",
		_group_anagrams,
		[(["eat", "tea", "tan", "ate", "nat", "bat"],), ([],), (["a"],)],
	),
	FunctionSpec(
		"top_k_frequent",
		"(values: list[int], k: int) -> list[int]",
		"Return the k most frequent values, most frequent first. Break ties by the smaller "
		"value first.",
		_top_k,
		[([1, 1, 1, 2, 2, 3], 2), ([4, 4, 5, 5, 6], 2), ([7], 1), ([3, 1, 2], 3)],
	),
	FunctionSpec(
		"insert_position",
		"(values: list[int], target: int) -> int",
		"Given a list sorted in ascending order, return the leftmost index where target could "
		"be inserted while keeping the list sorted. Use binary search.",
		_insert_position,
		[([1, 3, 5, 6], 5), ([1, 3, 5, 6], 2), ([1, 3, 5, 6], 7), ([], 4), ([2, 2, 2], 2)],
	),
	FunctionSpec(
		"is_valid_isbn10",
		"(code: str) -> bool",
		"Validate an ISBN-10. Ignore hyphens. The result must be nine digits followed by a digit "
		"or X (worth 10), and the sum of each digit times its weight (10 down to 1) must be "
		"divisible by 11.",
		_isbn10,
		[("0-306-40615-2",), ("0306406153",), ("123456789X",), ("12345",)],
	),
	FunctionSpec(
		"passes_luhn",
		"(number: str) -> bool",
		"Return True when a card number passes the Luhn checksum. The string may contain "
		"digits and spaces only; any other character, or fewer than two digits, makes it "
		"invalid.",
		_luhn,
		[("4539 3195 0343 6467",), ("8273 1232 7352 0569",), ("0",), ("059a",), ("059",)],
	),
	FunctionSpec(
		"format_duration",
		"(seconds: int) -> str",
		'Format a non-negative number of seconds as hours, minutes, and seconds, such as "1h 2m '
		'5s". Omit zero components, and return "0s" for zero.',
		_format_duration,
		[(3725,), (0,), (60,), (7200,), (59,)],
	),
	FunctionSpec(
		"parse_pairs",
		"(text: str) -> dict[str, str]",
		'Parse a string of key=value pairs separated by semicolons, such as "a=1; b = 2". Strip '
		"whitespace around keys and values, skip segments without an equals sign, and let later "
		"keys overwrite earlier ones. Only the first equals sign in a segment separates key and "
		"value.",
		_parse_pairs,
		[("a=1; b = 2",), ("",), ("x=1;;y;z=a=b",), ("k=1;k=2",)],
	),
	FunctionSpec(
		"longest_common_prefix",
		"(words: list[str]) -> str",
		"Return the longest string that every word in the list starts with. Return an empty "
		"string for an empty list.",
		_common_prefix,
		[(["flower", "flow", "flight"],), (["dog", "car"],), ([],), (["same", "same"],)],
	),
	FunctionSpec(
		"pascal_row",
		"(n: int) -> list[int]",
		"Return row n of Pascal's triangle, where row 0 is [1].",
		_pascal_row,
		[(0,), (1,), (4,), (6,)],
	),
	FunctionSpec(
		"collatz_steps",
		"(n: int) -> int",
		"Count the steps needed for a positive integer to reach 1 under the Collatz rule: halve "
		"even numbers, and replace odd numbers with 3n + 1.",
		_collatz,
		[(1,), (6,), (27,), (16,)],
	),
	FunctionSpec(
		"digital_root",
		"(n: int) -> int",
		"Repeatedly replace a non-negative integer with the sum of its digits until one digit "
		"remains, and return that digit.",
		_digital_root,
		[(0,), (38,), (99999,), (7,)],
	),
	FunctionSpec(
		"mask_email",
		"(email: str) -> str",
		"Mask the local part of an email address, keeping only its first character followed by "
		'"***". For example, "jane@example.com" becomes "j***@example.com".',
		_mask_email,
		[("jane@example.com",), ("a@b.co",), ("first.last@mail.example.org",)],
	),
	FunctionSpec(
		"title_case",
		"(text: str, minor: list[str]) -> str",
		"Capitalize the first letter of every word and lowercase the rest, except words listed "
		"in `minor`, which stay lowercase unless they are the first word. Words are separated by "
		"whitespace, and the result joins them with single spaces.",
		_title_case,
		[("the lord OF the rings", ["of", "the"]), ("a tale", ["a"]), ("", ["x"])],
	),
	FunctionSpec(
		"find_duplicates",
		"(values: list[int]) -> list[int]",
		"Return, in ascending order, each value that appears more than once in the list.",
		_duplicates,
		[([3, 1, 3, 2, 1],), ([],), ([5, 6, 7],), ([2, 2, 2],)],
	),
	FunctionSpec(
		"interleave",
		"(first: list, second: list) -> list",
		"Alternate elements from the two lists, starting with the first list. When one list runs "
		"out, append the rest of the other.",
		_interleave,
		[([1, 2, 3], ["a", "b"]), ([], [1]), ([1], [])],
	),
	FunctionSpec(
		"compress_ranges",
		"(values: list[int]) -> list[str]",
		"Given distinct integers in ascending order, describe consecutive runs as strings: a "
		'single value as "5" and a run as "1-3".',
		_compress_ranges,
		[([1, 2, 3, 5, 7, 8],), ([],), ([4],), ([-3, -2, 0],)],
	),
	FunctionSpec(
		"expand_ranges",
		"(spec: str) -> list[int]",
		'Expand a comma-separated range specification such as "1-3,5" into [1, 2, 3, 5]. '
		"Ignore whitespace and empty items. A leading minus sign marks a negative single value.",
		_expand_ranges,
		[("1-3,5",), ("",), ("10, 12-13",), ("-4,2",)],
	),
	FunctionSpec(
		"matrix_multiply",
		"(a: list[list[int]], b: list[list[int]]) -> list[list[int]]",
		"Multiply two matrices with compatible dimensions and return the product.",
		_matmul,
		[([[1, 2], [3, 4]], [[5, 6], [7, 8]]), ([[1, 2, 3]], [[1], [2], [3]]), ([[2]], [[3]])],
	),
	FunctionSpec(
		"gcd_of_list",
		"(values: list[int]) -> int",
		"Return the greatest common divisor of all integers in a non-empty list.",
		_gcd_list,
		[([12, 18, 24],), ([7],), ([100, 75, 50],), ([9, 28],)],
	),
	FunctionSpec(
		"is_anagram",
		"(first: str, second: str) -> bool",
		"Return True when the two strings contain the same letters with the same counts, "
		"ignoring spaces and case.",
		_is_anagram,
		[("Dormitory", "Dirty room"), ("abc", "abd"), ("", ""), ("aab", "ab")],
	),
	FunctionSpec(
		"remove_adjacent_duplicates",
		"(text: str) -> str",
		'Collapse each run of identical adjacent characters to one character, so "aabbcca" '
		'becomes "abca".',
		_remove_adjacent,
		[("aabbcca",), ("",), ("abc",), ("    x",)],
	),
	FunctionSpec(
		"days_between",
		"(first: str, second: str) -> int",
		"Return the absolute number of days between two dates written as YYYY-MM-DD.",
		_days_between,
		[("2024-01-01", "2024-03-01"), ("2023-12-31", "2023-12-25"), ("2020-02-28", "2021-02-28")],
	),
	FunctionSpec(
		"weighted_average",
		"(values: list[float], weights: list[float]) -> float",
		"Return the weighted average of the values, rounded to 3 decimal places. Both lists have "
		"the same length and the weights sum to a positive number.",
		_weighted_average,
		[([1, 2, 3], [1, 1, 1]), ([90, 70], [3, 1]), ([2.5], [4])],
	),
	FunctionSpec(
		"leaderboard",
		"(scores: dict[str, int]) -> list[str]",
		"Return the player names ordered by score from highest to lowest, breaking ties "
		"alphabetically.",
		_leaderboard,
		[({"ana": 5, "bo": 9, "cy": 5},), ({},), ({"z": 1},)],
	),
	FunctionSpec(
		"smallest_mode",
		"(values: list[int]) -> int",
		"Return the most frequent value in a non-empty list. When several values share the "
		"highest count, return the smallest of them.",
		_mode,
		[([1, 2, 2, 3, 3],), ([5],), ([4, 4, 1, 1, 0],)],
	),
	FunctionSpec(
		"hamming_distance",
		"(first: str, second: str) -> int",
		"Return the number of positions at which two equal-length strings differ.",
		lambda first, second: sum(a != b for a, b in zip(first, second, strict=True)),
		[("karolin", "kathrin"), ("", ""), ("1011101", "1001001")],
	),
	FunctionSpec(
		"to_binary",
		"(n: int, width: int) -> str",
		"Return the binary representation of a non-negative integer, left-padded with zeros to "
		"at least `width` characters.",
		lambda n, width: format(n, "b").zfill(width),
		[(5, 8), (0, 1), (255, 4), (9, 0)],
	),
	FunctionSpec(
		"parse_csv_line",
		"(line: str) -> list[str]",
		"Split one CSV line on commas. A field wrapped in double quotes may contain commas; the "
		"quotes are removed. Fields are not stripped, and an empty line yields one empty field.",
		_parse_csv_line,
		[('a,"b,c",d',), ("",), ('"x",,y',), ("one",)],
	),
	FunctionSpec(
		"max_paren_depth",
		"(text: str) -> int",
		"Return the maximum nesting depth of parentheses in the text. Assume they are balanced.",
		_max_depth,
		[("(a(b)(c(d)))",), ("",), ("()()",), ("x",)],
	),
	FunctionSpec(
		"star_histogram",
		"(counts: dict[str, int]) -> str",
		'Render a text histogram with one line per key in alphabetical order, formatted as "key: '
		'***" with one asterisk per count. Join lines with newline characters.',
		_histogram,
		[({"b": 2, "a": 3},), ({},), ({"x": 0},)],
	),
	FunctionSpec(
		"kth_smallest",
		"(values: list[int], k: int) -> int",
		"Return the k-th smallest element of the list, where k = 1 is the minimum. Duplicates "
		"count separately.",
		_kth_smallest,
		[([7, 10, 4, 3, 20, 15], 3), ([1], 1), ([5, 5, 1], 2)],
	),
	FunctionSpec(
		"snake_to_camel",
		"(name: str) -> str",
		'Convert a snake_case identifier to lowerCamelCase, so "max_retry_count" becomes '
		'"maxRetryCount".',
		_snake_to_camel,
		[("max_retry_count",), ("x",), ("http_status_code",)],
	),
	FunctionSpec(
		"camel_to_snake",
		"(name: str) -> str",
		"Convert a camelCase or PascalCase identifier to snake_case, inserting an underscore "
		'before each uppercase letter except the first character. "parseHttpBody" becomes '
		'"parse_http_body".',
		_camel_to_snake,
		[("parseHttpBody",), ("Value",), ("x",)],
	),
	FunctionSpec(
		"pairwise_differences",
		"(values: list[int]) -> list[int]",
		"Return the differences between consecutive elements, values[i + 1] - values[i].",
		lambda values: [b - a for a, b in zip(values, values[1:], strict=False)],
		[([1, 4, 9, 16],), ([5],), ([],), ([3, 1, 2],)],
	),
	FunctionSpec(
		"count_long_words",
		"(text: str, length: int) -> int",
		"Count the whitespace-separated words whose length, after removing leading and trailing "
		"punctuation characters .,;:!?, is strictly greater than `length`.",
		lambda text, length: sum(len(word.strip(".,;:!?")) > length for word in text.split()),
		[("Hello there, magnificent world!", 5), ("", 1), ("a bb ccc", 2)],
	),
	FunctionSpec(
		"lcm",
		"(a: int, b: int) -> int",
		"Return the least common multiple of two positive integers.",
		lambda a, b: a * b // math.gcd(a, b),
		[(4, 6), (7, 3), (12, 12), (1, 9)],
	),
]


def _signature_docstring(spec: FunctionSpec) -> str:
	examples = []
	for case in spec.cases[:2]:
		arguments = ", ".join(repr(argument) for argument in case)
		examples.append(f"\t>>> {spec.name}({arguments})\n\t{spec.implementation(*case)!r}")
	body = "\n".join(examples)
	return f'def {spec.name}{spec.signature}:\n\t"""{spec.description}\n\n{body}\n\t"""\n'


def _tests(spec: FunctionSpec) -> str:
	lines = []
	for case in spec.cases:
		arguments = ", ".join(repr(argument) for argument in case)
		lines.append(f"assert {spec.name}({arguments}) == {spec.implementation(*case)!r}")
	return "\n".join(lines)


def python_functions() -> Iterator[Generated]:
	"""Two prompt styles per function, matching the HumanEval and MBPP evaluation prompts."""
	for spec in FUNCTION_SPECS:
		tests = _tests(spec)
		reference = {"kind": "python_tests", "tests": tests, "entry_point": spec.name}
		yield Generated(
			key=f"{spec.name}-signature",
			group=f"python-{spec.name}",
			task="python-function",
			user=HUMANEVAL_PREFIX + _signature_docstring(spec),
			reference=reference,
		)
		visible = "\n".join(tests.splitlines()[:3])
		yield Generated(
			key=f"{spec.name}-tests",
			group=f"python-{spec.name}",
			task="python-function",
			user=MBPP_PREFIX
			+ f"Write a function `{spec.name}{spec.signature}`. {spec.description}"
			+ f"\n\nYour code must satisfy:\n{visible}",
			reference=reference,
		)


# Arithmetic word problems. Each template draws its own numbers and computes the answer.

NAMES = [
	"Amara",
	"Bilal",
	"Chen",
	"Dalia",
	"Emeka",
	"Farah",
	"Gustavo",
	"Hana",
	"Ines",
	"Jonah",
	"Kiri",
	"Lucia",
	"Mateo",
	"Nadia",
	"Oskar",
	"Priya",
	"Quinn",
	"Rosa",
	"Sami",
	"Tomas",
	"Uma",
	"Viktor",
	"Wen",
	"Ximena",
	"Yusuf",
	"Zoe",
]

type MathTemplate = Callable[[random.Random], tuple[str, float]]


def _money(value: float) -> float:
	return round(value, 2)


def _bakery(rng: random.Random) -> tuple[str, float]:
	trays, per_tray = rng.randint(3, 12), rng.randint(6, 24)
	total = trays * per_tray
	morning = rng.randint(1, total // 2)
	afternoon = rng.randint(0, total - morning)
	name = rng.choice(NAMES)
	return (
		f"{name}'s bakery bakes {trays} trays of muffins with {per_tray} muffins on each tray. "
		f"It sells {morning} muffins in the morning and {afternoon} in the afternoon. "
		"How many muffins are left at the end of the day?",
		total - morning - afternoon,
	)


def _discount(rng: random.Random) -> tuple[str, float]:
	price, count = rng.choice([2.5, 3, 4.75, 6, 8.2, 12]), rng.randint(2, 15)
	percent = rng.choice([10, 15, 20, 25, 30])
	name = rng.choice(NAMES)
	return (
		f"Notebooks cost ${price} each. {name} buys {count} notebooks and gets a {percent}% "
		f"discount on the whole purchase. How many dollars does {name} pay?",
		_money(price * count * (100 - percent) / 100),
	)


def _two_leg_trip(rng: random.Random) -> tuple[str, float]:
	speed1, hours1 = rng.randint(40, 120), rng.randint(1, 5)
	speed2, hours2 = rng.randint(30, 110), rng.randint(1, 4)
	return (
		f"A train travels at {speed1} km per hour for {hours1} hours, then at {speed2} km per "
		f"hour for {hours2} more hours. How many kilometers does it travel in total?",
		speed1 * hours1 + speed2 * hours2,
	)


def _fence(rng: random.Random) -> tuple[str, float]:
	length, width, cost = rng.randint(5, 60), rng.randint(3, 40), rng.choice([4, 7, 9, 12, 15])
	return (
		f"A rectangular garden is {length} meters long and {width} meters wide. Fencing costs "
		f"${cost} per meter. How many dollars does it cost to fence the whole perimeter?",
		2 * (length + width) * cost,
	)


def _savings(rng: random.Random) -> tuple[str, float]:
	start, weekly, weeks = rng.randint(20, 300), rng.randint(5, 60), rng.randint(3, 20)
	spent = rng.randint(0, start + weekly * weeks)
	name = rng.choice(NAMES)
	return (
		f"{name} has ${start} saved and adds ${weekly} every week for {weeks} weeks, then "
		f"spends ${spent} on a bicycle. How many dollars does {name} have left?",
		start + weekly * weeks - spent,
	)


def _recipe(rng: random.Random) -> tuple[str, float]:
	servings = rng.choice([2, 4, 6, 8])
	cups = rng.randint(1, 6)
	target = servings * rng.randint(2, 6)
	return (
		f"A recipe uses {cups} cups of flour to make {servings} servings. How many cups of flour "
		f"are needed for {target} servings?",
		cups * target // servings,
	)


def _ages(rng: random.Random) -> tuple[str, float]:
	younger, gap, years = rng.randint(3, 40), rng.randint(1, 30), rng.randint(1, 15)
	first, second = rng.sample(NAMES, 2)
	total = 2 * younger + gap + 2 * years
	return (
		f"{first} is {gap} years older than {second}. In {years} years, the sum of their ages "
		f"will be {total}. How old is {second} now?",
		younger,
	)


def _tank(rng: random.Random) -> tuple[str, float]:
	drain = rng.randint(1, 8)
	net = rng.randint(2, 10)
	minutes = rng.randint(5, 40)
	return (
		f"A tank holds {net * minutes} liters. Water flows in at {drain + net} liters per minute "
		f"while a leak drains {drain} liters per minute. Starting empty, how many minutes does "
		"the tank take to fill?",
		minutes,
	)


def _percent_growth(rng: random.Random) -> tuple[str, float]:
	january = rng.randint(2, 40) * 20
	percent = rng.choice([5, 10, 15, 20, 25, 40, 50])
	return (
		f"A shop sold {january} lamps in January and {percent}% more lamps in February than in "
		"January. How many lamps did it sell in the two months combined?",
		january + january * (100 + percent) // 100,
	)


def _tickets(rng: random.Random) -> tuple[str, float]:
	adult, child = rng.randint(8, 30), rng.randint(4, 15)
	adults, children = rng.randint(1, 4), rng.randint(1, 5)
	cost = adult * adults + child * children
	paid = (cost // 50 + 1) * 50
	return (
		f"Museum tickets cost ${adult} for adults and ${child} for children. A family buys "
		f"{adults} adult and {children} child tickets and pays with ${paid}. How many dollars "
		"of change do they receive?",
		paid - cost,
	)


def _overtime(rng: random.Random) -> tuple[str, float]:
	wage, hours = rng.choice([12, 15, 18, 20, 22, 24]), rng.randint(30, 55)
	name = rng.choice(NAMES)
	pay = wage * min(hours, 40) + wage * 1.5 * max(hours - 40, 0)
	return (
		f"{name} earns ${wage} per hour and 1.5 times that rate for every hour over 40 in a week. "
		f"How many dollars does {name} earn in a week of {hours} hours?",
		_money(pay),
	)


def _phone_plan(rng: random.Random) -> tuple[str, float]:
	base, extra, included = rng.randint(15, 45), rng.choice([2, 3, 5, 8]), rng.randint(2, 20)
	used = included + rng.randint(-2, 12)
	return (
		f"A phone plan costs ${base} per month and includes {included} GB of data. Each extra "
		f"GB costs ${extra}. What is the monthly bill in dollars for using {max(used, 1)} GB?",
		base + extra * max(max(used, 1) - included, 0),
	)


def _needed_score(rng: random.Random) -> tuple[str, float]:
	count, target = rng.randint(2, 5), rng.randint(70, 90)
	scores = [rng.randint(55, 100) for _ in range(count)]
	required = target * (count + 1) - sum(scores)
	while not 0 <= required <= 100:
		scores = [rng.randint(55, 100) for _ in range(count)]
		required = target * (count + 1) - sum(scores)
	listing = ", ".join(str(score) for score in scores)
	return (
		f"A student scored {listing} on {count} tests. What score is needed on the next test for "
		f"the average over all {count + 1} tests to be exactly {target}?",
		required,
	)


def _meeting(rng: random.Random) -> tuple[str, float]:
	speed1, speed2, hours = rng.randint(8, 30), rng.randint(8, 30), rng.randint(1, 6)
	return (
		f"Two cyclists start {(speed1 + speed2) * hours} km apart and ride toward each other, "
		f"one at {speed1} km per hour and the other at {speed2} km per hour. After how many "
		"hours do they meet?",
		hours,
	)


def _laps(rng: random.Random) -> tuple[str, float]:
	laps, track, days = rng.randint(2, 12), rng.choice([200, 250, 400, 500]), rng.randint(2, 14)
	return (
		f"A runner completes {laps} laps of a {track}-meter track every day for {days} days. How "
		"many kilometers does the runner cover in total?",
		laps * track * days / 1000,
	)


def _boxes(rng: random.Random) -> tuple[str, float]:
	per_box, items = rng.randint(6, 48), rng.randint(50, 900)
	return (
		f"Each box holds {per_box} jars. How many boxes are needed to pack {items} jars?",
		-(-items // per_box),
	)


def _electricity(rng: random.Random) -> tuple[str, float]:
	watts, hours, days = (
		rng.choice([100, 250, 500, 1000, 1500]),
		rng.randint(1, 10),
		rng.randint(5, 31),
	)
	cents = rng.choice([10, 12, 15, 20, 25])
	return (
		f"A {watts}-watt heater runs {hours} hours a day for {days} days. Electricity costs "
		f"{cents} cents per kilowatt-hour. How many dollars does it cost to run?",
		_money(watts * hours * days / 1000 * cents / 100),
	)


def _pizza(rng: random.Random) -> tuple[str, float]:
	slices, people, each = rng.choice([6, 8, 10, 12]), rng.randint(4, 30), rng.randint(1, 4)
	return (
		f"Each pizza is cut into {slices} slices. {people} people each eat {each} slices. What is "
		"the smallest number of pizzas needed?",
		-(-(people * each) // slices),
	)


def _shares(rng: random.Random) -> tuple[str, float]:
	parts = rng.sample(range(1, 7), 3)
	unit = rng.randint(5, 60)
	first, second, third = rng.sample(NAMES, 3)
	return (
		f"{first}, {second}, and {third} split ${sum(parts) * unit} in the ratio "
		f"{parts[0]}:{parts[1]}:{parts[2]}. How many dollars does {third} receive?",
		parts[2] * unit,
	)


def _paint(rng: random.Random) -> tuple[str, float]:
	walls, area, coverage = rng.randint(2, 8), rng.randint(8, 30), rng.choice([10, 12, 15])
	coats = rng.randint(1, 3)
	return (
		f"A room has {walls} walls of {area} square meters each. Each can of paint covers "
		f"{coverage} square meters, and the walls need {coats} coats. How many whole cans must "
		"be bought?",
		-(-(walls * area * coats) // coverage),
	)


def _clock(rng: random.Random) -> tuple[str, float]:
	start_hour, start_minute = rng.randint(6, 11), rng.choice([0, 10, 15, 20, 30, 45])
	tasks = [rng.randint(10, 75) for _ in range(rng.randint(2, 4))]
	listing = ", ".join(f"{t} minutes" for t in tasks)
	return (
		f"A worker starts at {start_hour}:{start_minute:02d} in the morning and does tasks taking "
		f"{listing}, with no breaks. How many minutes after {start_hour}:00 does the last task "
		"finish?",
		start_minute + sum(tasks),
	)


MATH_TEMPLATES: list[tuple[str, MathTemplate]] = [
	("bakery", _bakery),
	("discount", _discount),
	("two-leg-trip", _two_leg_trip),
	("fence", _fence),
	("savings", _savings),
	("recipe", _recipe),
	("ages", _ages),
	("tank", _tank),
	("percent-growth", _percent_growth),
	("tickets", _tickets),
	("overtime", _overtime),
	("phone-plan", _phone_plan),
	("needed-score", _needed_score),
	("meeting", _meeting),
	("laps", _laps),
	("boxes", _boxes),
	("electricity", _electricity),
	("pizza", _pizza),
	("shares", _shares),
	("paint", _paint),
	("clock", _clock),
]


def _number(value: float) -> str:
	text = f"{value:.2f}".rstrip("0").rstrip(".")
	return "0" if text == "-0" else text


def word_problems(count: int, seed: int) -> Iterator[Generated]:
	rng = random.Random(f"{seed}:word-problems")
	seen: set[str] = set()
	attempts = 0
	while len(seen) < count and attempts < count * 20:
		attempts += 1
		name, template = MATH_TEMPLATES[attempts % len(MATH_TEMPLATES)]
		question, answer = template(rng)
		if question in seen or answer < 0:
			continue
		seen.add(question)
		yield Generated(
			key=f"{name}-{len(seen):05d}",
			group=f"math-{name}",
			task="math-word-problem",
			user=question + GSM8K_SUFFIX,
			reference={"kind": "numeric", "answer": _number(answer)},
		)


# Structured extraction. The prose states each field once; the answer is the exact object.

CITIES = [
	"Lisbon",
	"Nairobi",
	"Osaka",
	"Toronto",
	"Valparaiso",
	"Krakow",
	"Hanoi",
	"Adelaide",
	"Accra",
	"Tbilisi",
	"Montevideo",
	"Bergen",
	"Pune",
	"Cork",
	"Tucson",
	"Ljubljana",
]
JOBS = [
	"pharmacist",
	"civil engineer",
	"data analyst",
	"carpenter",
	"translator",
	"nurse",
	"marine biologist",
	"accountant",
	"pilot",
	"librarian",
	"chef",
	"electrician",
]
PRODUCTS = [
	"desk lamp",
	"water bottle",
	"wireless mouse",
	"yoga mat",
	"kettle",
	"backpack",
	"notebook set",
	"phone stand",
	"camping stove",
	"rain jacket",
	"chess set",
	"toaster",
]
SURNAMES = [
	"Okafor",
	"Lindqvist",
	"Moreau",
	"Tanaka",
	"Haddad",
	"Novak",
	"Pereira",
	"Kowalski",
	"Mensah",
	"Iyer",
	"Brennan",
	"Castillo",
	"Fischer",
	"Abara",
]
VENUES = [
	"Riverside Hall",
	"the Old Mill",
	"Harbor Pavilion",
	"Room 204",
	"the Civic Theater",
	"Northgate Library",
	"Maple Community Center",
	"the Glasshouse",
]
EVENTS = [
	"poetry reading",
	"robotics workshop",
	"charity run briefing",
	"jazz evening",
	"budgeting seminar",
	"board game night",
	"film screening",
	"pottery class",
]
MONTHS = [
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December",
]


def _article(word: str) -> str:
	return "an" if word[0] in "aeiou" else "a"


def _contact(rng: random.Random) -> tuple[str, dict[str, Any], str]:
	first, last = rng.choice(NAMES), rng.choice(SURNAMES)
	age, city, job = rng.randint(21, 74), rng.choice(CITIES), rng.choice(JOBS)
	email = f"{first.lower()}.{last.lower()}@example.org"
	text = rng.choice(
		[
			f"{first} {last} is a {age}-year-old {job} living in {city}. You can reach "
			f"{first} at {email}.",
			f"Meet our new member: {first} {last}, {age}, works as {_article(job)} {job}. "
			f"Based in {city}. Contact: {email}",
			f"Email {email} to get in touch with {first} {last}, {_article(job)} {job} from "
			f"{city} who turned {age} this year.",
		]
	)
	return (
		text,
		{"name": f"{first} {last}", "age": age, "city": city, "occupation": job, "email": email},
		"name (string), age (integer), city (string), occupation (string), email (string)",
	)


def _order(rng: random.Random) -> tuple[str, dict[str, Any], str]:
	order_id = f"ORD-{rng.randint(10000, 99999)}"
	product, quantity = rng.choice(PRODUCTS), rng.randint(1, 9)
	unit = rng.choice([4.5, 9.99, 12.0, 15.25, 23.4, 39.0, 64.5])
	text = rng.choice(
		[
			f"Order {order_id} contains {quantity} units of the {product} at ${unit:.2f} each.",
			f"We have shipped your {product} (quantity {quantity}, ${unit:.2f} per unit). "
			f"Reference number: {order_id}.",
			f"Thanks for buying {quantity} x {product}. Each one cost ${unit:.2f}. "
			f"Keep {order_id} for returns.",
		]
	)
	return (
		text,
		{
			"order_id": order_id,
			"product": product,
			"quantity": quantity,
			"unit_price": unit,
			"total": round(unit * quantity, 2),
		},
		"order_id (string), product (string), quantity (integer), unit_price (number), "
		"total (number, quantity times unit_price rounded to 2 decimals)",
	)


def _event(rng: random.Random) -> tuple[str, dict[str, Any], str]:
	event, venue = rng.choice(EVENTS), rng.choice(VENUES)
	day = date(rng.randint(2024, 2027), rng.randint(1, 12), rng.randint(1, 28))
	capacity = rng.randint(2, 40) * 10
	spoken = f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
	text = rng.choice(
		[
			f"Join us for a {event} at {venue} on {spoken}. Seats are limited to {capacity}.",
			f"On {spoken}, {venue} hosts a {event}. Up to {capacity} guests can attend.",
			f"Capacity {capacity}. The {event} takes place at {venue}; the date is {spoken}.",
		]
	)
	return (
		text,
		{"event": event, "venue": venue, "date": day.isoformat(), "capacity": capacity},
		"event (string), venue (string), date (string in YYYY-MM-DD format), capacity (integer)",
	)


EXTRACTORS = [("contact", _contact), ("order", _order), ("event", _event)]


def json_extraction(count: int, seed: int) -> Iterator[Generated]:
	rng = random.Random(f"{seed}:json-extraction")
	seen: set[str] = set()
	attempts = 0
	while len(seen) < count and attempts < count * 20:
		attempts += 1
		name, extractor = EXTRACTORS[attempts % len(EXTRACTORS)]
		text, expected, keys = extractor(rng)
		if text in seen:
			continue
		seen.add(text)
		yield Generated(
			key=f"{name}-{len(seen):05d}",
			group=f"json-{name}",
			task="json-extraction",
			system="Return only valid JSON.",
			user=f"Extract these fields from the text: {keys}.\n\nText: {text}",
			reference={"kind": "json_equals", "answer": json.dumps(expected, sort_keys=True)},
		)


# Exact-answer text and list operations.

WORDS = [
	"anchor",
	"basil",
	"canyon",
	"dynamo",
	"ember",
	"falcon",
	"glacier",
	"harbor",
	"indigo",
	"juniper",
	"kelp",
	"lantern",
	"meadow",
	"nebula",
	"orchid",
	"pepper",
	"quartz",
	"raven",
	"saffron",
	"tundra",
	"umber",
	"velvet",
	"willow",
	"xylem",
	"yarrow",
	"zephyr",
]


def _sort_numbers(rng: random.Random) -> tuple[str, str]:
	values = rng.sample(range(-50, 500), rng.randint(5, 10))
	listing = ", ".join(map(str, values))
	return (
		f"Sort these numbers in ascending order: {listing}. Return only the sorted numbers "
		"separated by commas.",
		", ".join(map(str, sorted(values))),
	)


def _alphabetize(rng: random.Random) -> tuple[str, str]:
	words = rng.sample(WORDS, rng.randint(4, 8))
	return (
		f"Put these words in alphabetical order: {', '.join(words)}. Return only the words "
		"separated by commas.",
		", ".join(sorted(words)),
	)


def _reformat_date(rng: random.Random) -> tuple[str, str]:
	day = date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
	return (
		f"Rewrite the date {day.isoformat()} in the format 'day month-name year', for example "
		"'7 March 2021'. Return only the rewritten date.",
		f"{day.day} {MONTHS[day.month - 1]} {day.year}",
	)


def _convert_units(rng: random.Random) -> tuple[str, str]:
	options = [
		("kilometers", "meters", 1000),
		("hours", "minutes", 60),
		("kilograms", "grams", 1000),
		("liters", "milliliters", 1000),
		("days", "hours", 24),
		("feet", "inches", 12),
	]
	source, target, factor = rng.choice(options)
	value = round(rng.uniform(0.1, 100), rng.choice([0, 1, 2])) or 1
	return (
		f"Convert {value} {source} to {target}. Return only the number.",
		_number(value * factor),
	)


def _reverse_words(rng: random.Random) -> tuple[str, str]:
	words = rng.sample(WORDS, rng.randint(4, 7))
	return (
		f'Reverse the order of the words in this phrase: "{" ".join(words)}". Return only the '
		"reversed phrase without quotes.",
		" ".join(reversed(words)),
	)


def _dedupe(rng: random.Random) -> tuple[str, str]:
	words = [rng.choice(WORDS[:10]) for _ in range(rng.randint(6, 12))]
	return (
		f"Remove duplicates from this list while keeping the first occurrence of each item in "
		f"its original order: {', '.join(words)}. Return only the resulting items separated by "
		"commas.",
		", ".join(dict.fromkeys(words)),
	)


def _extract_emails(rng: random.Random) -> tuple[str, str]:
	people = rng.sample(NAMES, rng.randint(2, 4))
	emails = [f"{name.lower()}{rng.randint(1, 99)}@example.net" for name in people]
	sentences = [
		f"{name} can be reached at {email}." for name, email in zip(people, emails, strict=True)
	]
	return (
		"List every email address in the following text in order of appearance, separated by "
		"commas, with no other text.\n\n" + " Meanwhile, ".join(sentences),
		", ".join(emails),
	)


def _largest(rng: random.Random) -> tuple[str, str]:
	values = [
		round(rng.uniform(-100, 100), rng.choice([0, 1, 2])) for _ in range(rng.randint(5, 9))
	]
	listing = ", ".join(_number(v) for v in values)
	return (
		f"Which is the largest of these numbers: {listing}? Return only that number.",
		_number(max(values)),
	)


TEXT_OPERATIONS = [
	("sort-numbers", _sort_numbers),
	("alphabetize", _alphabetize),
	("reformat-date", _reformat_date),
	("convert-units", _convert_units),
	("reverse-words", _reverse_words),
	("dedupe", _dedupe),
	("extract-emails", _extract_emails),
	("largest", _largest),
]


def text_operations(count: int, seed: int) -> Iterator[Generated]:
	rng = random.Random(f"{seed}:text-operations")
	seen: set[str] = set()
	attempts = 0
	while len(seen) < count and attempts < count * 20:
		attempts += 1
		name, operation = TEXT_OPERATIONS[attempts % len(TEXT_OPERATIONS)]
		question, answer = operation(rng)
		if question in seen:
			continue
		seen.add(question)
		yield Generated(
			key=f"{name}-{len(seen):05d}",
			group=f"text-{name}",
			task="exact-operation",
			user=question,
			reference={"kind": "exact", "answer": answer},
		)


# Instruction following with measurable format constraints over a broad topic list.

TOPICS = [
	"how vaccines train the immune system",
	"the water cycle",
	"compound interest",
	"how a bill becomes law in a parliamentary system",
	"photosynthesis",
	"plate tectonics",
	"how public-key cryptography works",
	"the causes of inflation",
	"how airplanes generate lift",
	"the role of mitochondria",
	"supply and demand",
	"how search engines rank pages",
	"the printing press and its effects",
	"why the seasons change",
	"how batteries store energy",
	"the basics of version control",
	"how a refrigerator keeps food cold",
	"black holes",
	"the scientific method",
	"how credit scores work",
	"the history of the bicycle",
	"how coral reefs form",
	"how noise-cancelling headphones work",
	"the rules of chess",
	"how to start a vegetable garden",
	"the difference between weather and climate",
	"how the heart pumps blood",
	"how compilers turn code into programs",
	"the Silk Road",
	"recycling plastics",
	"how GPS determines location",
	"the importance of sleep",
	"how bees communicate",
	"the basics of personal budgeting",
	"how earthquakes are measured",
	"how a bill of materials is used in manufacturing",
	"the Industrial Revolution",
	"how solar panels produce electricity",
	"the structure of DNA",
	"how to prepare for a job interview",
	"what an API is",
	"how tides work",
	"the basics of probability",
	"how to write a cover letter",
	"how wind turbines work",
	"the life cycle of a star",
	"how antibiotics work",
	"the difference between RAM and storage",
	"how to give constructive feedback",
	"the history of the Olympic Games",
	"how volcanoes form",
	"how elections are audited",
	"the basics of nutrition labels",
	"how sourdough bread rises",
	"what machine learning is",
	"how a mortgage works",
	"the role of enzymes in digestion",
	"how rainbows form",
	"time management techniques",
	"how the internet routes data",
	"why leaves change color",
	"the basics of first aid for minor burns",
	"how a camera lens focuses light",
	"what a database index does",
	"how glaciers shape landscapes",
	"the Renaissance",
	"how to care for a houseplant",
	"how electric motors work",
	"the basics of composting",
	"how interest rates affect borrowing",
	"the structure of an atom",
	"how to run a meeting well",
	"the purpose of unit testing",
	"how soap removes grease",
	"the history of writing systems",
	"how a thermostat works",
	"how languages borrow words",
	"the basics of map projections",
	"how hybrid cars save fuel",
	"the human digestive system",
	"how to read a weather map",
	"how microwaves heat food",
	"the purpose of a central bank",
	"how fireflies glow",
	"how to back up personal files",
	"how the ear processes sound",
	"the basics of supply chains",
	"how 3D printing works",
	"the importance of biodiversity",
	"how to negotiate a salary",
	"how caching speeds up websites",
	"the history of tea",
	"how a lock and key work",
	"what open-source software is",
	"the basics of meditation",
	"how clouds form",
	"how to plan a study schedule",
	"the basics of stock market indexes",
	"how bridges carry loads",
	"how the moon affects Earth",
	"how password managers work",
	"the origins of jazz",
]

AUDIENCES = ["ten-year-old", "busy executive", "first-year university student", "retiree"]
FILLER_WORDS = ["very", "really", "basically", "actually", "just"]


def constrained_writing(count: int, seed: int) -> Iterator[Generated]:
	rng = random.Random(f"{seed}:constrained-writing")
	combinations = [(topic, style) for topic in TOPICS for style in range(4)]
	rng.shuffle(combinations)
	for index, (topic, style) in enumerate(combinations[:count]):
		if style == 0:
			bullets = rng.randint(3, 6)
			user = (
				f"Write exactly {bullets} bullet points about {topic}. Start each bullet with "
				'"- " and write nothing else.'
			)
			constraints: dict[str, Any] = {"bullet_count": bullets}
		elif style == 1:
			limit = rng.choice([40, 60, 80, 100])
			user = f"Explain {topic} to a {rng.choice(AUDIENCES)} in no more than {limit} words."
			constraints = {"max_words": limit}
		elif style == 2:
			minimum, banned = rng.choice([120, 150, 200]), rng.choice(FILLER_WORDS)
			user = (
				f"Write an informative paragraph of at least {minimum} words about {topic}. Do "
				f'not use the word "{banned}".'
			)
			constraints = {"min_words": minimum, "must_exclude": [banned]}
		else:
			user = (
				f"Summarize {topic} in two short paragraphs, then end with a line that starts "
				'with "Key takeaway:".'
			)
			constraints = {"must_include": ["Key takeaway:"]}
		yield Generated(
			key=f"constrained-{index:05d}",
			group=f"constrained-{topic}",
			task="constrained-writing",
			user=user,
			reference={"kind": "general", "constraints": constraints},
		)
