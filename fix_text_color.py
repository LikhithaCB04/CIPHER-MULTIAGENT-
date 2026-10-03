with open("frontend/src/IDE.tsx", "r", encoding="utf-8") as f:
    content = f.read()

# Force text-white on textarea
content = content.replace('text-white placeholder-[#333]', '!text-white placeholder-[#666]')

with open("frontend/src/IDE.tsx", "w", encoding="utf-8") as f:
    f.write(content)
