import re

with open('frontend/src/IDE.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update accept list
content = content.replace('accept="image/*,text/*,application/json,text/markdown,.py,.js,.jsx,.ts,.tsx,.html,.css,.csv"', 'accept="image/*,text/*,application/json,text/markdown,.py,.js,.jsx,.ts,.tsx,.html,.css,.csv,.xlsx,.xls"')

# 2. Update handleFileChange
old_file_change = """        if (file.type.startsWith('image/')) {
          reader.readAsDataURL(file);
        } else {
          reader.readAsText(file);
        }"""
new_file_change = """        if (file.type.startsWith('image/') || file.name.toLowerCase().endsWith('.xlsx') || file.name.toLowerCase().endsWith('.xls')) {
          reader.readAsDataURL(file);
        } else {
          reader.readAsText(file);
        }"""
content = content.replace(old_file_change, new_file_change)

# 3. Update handleSend
old_send = """    try {
      const res = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: text, context: '' }),
      });"""
new_send = """    try {
      const contextData = attachments.length > 0 ? JSON.stringify({ files: attachments }) : '';
      setAttachments([]);
      
      const res = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: text, context: contextData }),
      });"""
content = content.replace(old_send, new_send)

# 4. Display result files
old_result = """                              {res.result && (
                                <pre className="bg-[#000] p-3 rounded-lg overflow-x-auto text-[10px] text-emerald-400 border border-[#1a1a1a] whitespace-pre-wrap break-words">
                                  <code>{res.result}</code>
                                </pre>
                              )}
                            </div>"""
new_result = """                              {res.result && (
                                <pre className="bg-[#000] p-3 rounded-lg overflow-x-auto text-[10px] text-emerald-400 border border-[#1a1a1a] whitespace-pre-wrap break-words">
                                  <code>{res.result}</code>
                                </pre>
                              )}
                              {res.files && res.files.length > 0 && (
                                <div className="mt-2 space-y-2">
                                  {res.files.map((f: any, fi: number) => (
                                    <div key={fi}>
                                      <a href={f.data} download={f.name} className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#111] border border-[#222] hover:bg-[#222] text-[#fff] text-xs rounded transition-colors">
                                        <FileText className="w-3 h-3" /> Download {f.name}
                                      </a>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>"""
content = content.replace(old_result, new_result)

with open('frontend/src/IDE.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
