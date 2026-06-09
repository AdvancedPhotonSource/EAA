const escapeHtml = (value: unknown): string =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const inlineMarkdown = (text: string): string => {
  let output = escapeHtml(text);
  const code: string[] = [];
  output = output.replace(/(`+)([\s\S]*?)\1/g, (_match, _ticks: string, value: string) => {
    const id = code.length;
    code.push(`<code>${escapeHtml(value)}</code>`);
    return `\u0000CODE${id}\u0000`;
  });
  output = output.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
  );
  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return output.replace(/\u0000CODE(\d+)\u0000/g, (_match, id: string) => code[Number(id)] ?? "");
};

const renderTable = (lines: string[]): string => {
  const rows = lines.map((line) =>
    line
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => inlineMarkdown(cell.trim())),
  );
  const header = rows[0] ?? [];
  const body = rows.slice(2);
  return `<table><thead><tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${body
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
};

export const renderMarkdown = (markdown: unknown): string => {
  const lines = String(markdown ?? "").replace(/\r\n/g, "\n").split("\n");
  const blocks: string[] = [];
  let paragraph: string[] = [];
  let code: string[] | null = null;
  let table: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${inlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
    paragraph = [];
  };
  const flushTable = () => {
    if (!table.length) return;
    blocks.push(renderTable(table));
    table = [];
  };
  const flushList = () => {
    if (!list) return;
    const tag = list.ordered ? "ol" : "ul";
    blocks.push(`<${tag}>${list.items.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    list = null;
  };

  for (const line of lines) {
    const fence = line.match(/^```|^~~~/);
    if (fence && code === null) {
      flushParagraph();
      flushTable();
      flushList();
      code = [];
      continue;
    }
    if (fence && code !== null) {
      blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      code = null;
      continue;
    }
    if (code !== null) {
      code.push(line);
      continue;
    }
    const unorderedItem = line.match(/^\s*[-*+]\s+(.+)$/);
    const orderedItem = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedItem || orderedItem) {
      flushParagraph();
      flushTable();
      const ordered = Boolean(orderedItem);
      if (!list || list.ordered !== ordered) flushList();
      if (!list) list = { ordered, items: [] };
      list.items.push((unorderedItem ?? orderedItem)?.[1] ?? "");
      continue;
    }
    if (/^\s*\|.+\|\s*$/.test(line)) {
      flushParagraph();
      flushList();
      table.push(line);
      continue;
    }
    flushTable();
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s*$/.test(line)) {
      flushParagraph();
      flushList();
      continue;
    }
    paragraph.push(line);
  }
  if (code !== null) blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushTable();
  flushList();
  flushParagraph();
  return blocks.join("\n");
};

export { escapeHtml };
