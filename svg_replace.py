import re

svg_code = """    <svg width="24" height="28" viewBox="0 0 28 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right:2px; flex-shrink:0;">
      <mask id="halfMask">
        <path d="M14 2 C6 3 2 9 2 16 C2 21 6 27 14 30 V2 Z" fill="white"/>
        <path d="M5 15 Q9 12 12 15 Q9 18 5 15" fill="black"/>
      </mask>
      <rect x="0" y="0" width="14" height="32" fill="currentColor" mask="url(#halfMask)" opacity="0.85"/>
      <circle cx="8.5" cy="15" r="1.5" fill="#187a52"/>
      <g stroke="#d4145a" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2 C22 3 26 9 26 16 C26 21 22 27 14 30"/>
        <line x1="14" y1="2" x2="14" y2="30"/>
        <path d="M16 15 Q20 12 24 15 Q20 18 16 15"/>
        <path d="M14 6 Q20 7 24 9"/>
        <path d="M14 10 Q21 11 25.5 14"/>
        <path d="M14 20 Q20 22 25 21"/>
        <path d="M14 25 Q18 26 22 26"/>
        <path d="M17 3 Q17 10 16 15 M16 15 Q17 22 17 28"/>
        <path d="M21 4 Q21 10 24 15 M24 15 Q23 21 20 27"/>
      </g>
      <rect x="23" y="3" width="2" height="2" fill="#d4145a"/>
      <rect x="26" y="5" width="2" height="2" fill="#d4145a"/>
      <rect x="21" y="6" width="1.5" height="1.5" fill="#d4145a"/>
      <rect x="27" y="9" width="1.5" height="1.5" fill="#d4145a"/>
      <rect x="18" y="25" width="2" height="2" fill="#d4145a"/>
    </svg>"""

with open('frontend.html', 'r', encoding='utf-8') as f:
    html = f.read()

html = re.sub(r'<img src="data:image/png;base64,.*?" alt="Logo".*?>', svg_code, html, flags=re.DOTALL)

with open('frontend.html', 'w', encoding='utf-8') as f:
    f.write(html)
