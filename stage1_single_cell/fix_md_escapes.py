"""还原 markdown 元字符转义：\\# -> #, \\* -> *, \\_ -> _ 等。"""
import io

p = r'E:\gamma_sar\docs\status_2026-10-11.md'
s = io.open(p, encoding='utf-8').read()

pairs = [
    ('\\#', '#'), ('\\*', '*'), ('\\_', '_'),
    ('\\[', '['), ('\\]', ']'),
    ('\\(', '('), ('\\)', ')'),
    ('\\>', '>'), ('\\|', '|'), ('\\-', '-'),
]
s2 = s
for a, b in pairs:
    s2 = s2.replace(a, b)

io.open(p, 'w', encoding='utf-8').write(s2)
print('before:', len(s), 'after:', len(s2), 'delta:', len(s) - len(s2))
print('head:')
print(s2[:200])