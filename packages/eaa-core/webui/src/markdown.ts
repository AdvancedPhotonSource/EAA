const escapeHtml = (value: unknown): string =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const mathPattern = /(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$)/g;

const protectMath = (text: string) => {
  const math: string[] = [];
  const protectedText = text.replace(mathPattern, (match) => {
    const id = math.length;
    math.push(match);
    return `\u0000MATH${id}\u0000`;
  });
  return { protectedText, math };
};

const restoreMath = (text: string, math: string[]) =>
  text.replace(/\u0000MATH(\d+)\u0000/g, (_match, id: string) => escapeHtml(math[Number(id)] ?? ""));

const inlineMarkdown = (text: string): string => {
  const { protectedText, math } = protectMath(text);
  let output = escapeHtml(protectedText);
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
  output = output.replace(/\u0000CODE(\d+)\u0000/g, (_match, id: string) => code[Number(id)] ?? "");
  return restoreMath(output, math);
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
  let displayMath: { end: string; lines: string[] } | null = null;
  let table: string[] = [];
  let list: { ordered: boolean; items: string[]; itemIndents: number[]; start: number | null } | null = null;

  const renderListItem = (item: string): string => {
    const rendered = renderMarkdown(item);
    const paragraphMatch = rendered.match(/^<p>([\s\S]*)<\/p>$/);
    return paragraphMatch ? paragraphMatch[1] : rendered;
  };
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
    const attrs = list.ordered && list.start !== null && list.start > 1 ? ` start="${list.start}"` : "";
    blocks.push(`<${tag}${attrs}>${list.items.map((item) => `<li>${renderListItem(item)}</li>`).join("")}</${tag}>`);
    list = null;
  };

  for (const line of lines) {
    if (displayMath !== null) {
      displayMath.lines.push(line);
      if (line.trim().endsWith(displayMath.end)) {
        blocks.push(`<div class="eaa-math-block">${escapeHtml(displayMath.lines.join("\n"))}</div>`);
        displayMath = null;
      }
      continue;
    }
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
    const trimmedLine = line.trim();
    if ((trimmedLine === "\\[" || trimmedLine.startsWith("\\[")) && !trimmedLine.endsWith("\\]")) {
      flushParagraph();
      flushTable();
      flushList();
      displayMath = { end: "\\]", lines: [line] };
      continue;
    }
    if (trimmedLine === "$$" || (trimmedLine.startsWith("$$") && !trimmedLine.endsWith("$$"))) {
      flushParagraph();
      flushTable();
      flushList();
      displayMath = { end: "$$", lines: [line] };
      continue;
    }
    if (/^\s*(\\\[[\s\S]*\\\]|\$\$[\s\S]*\$\$)\s*$/.test(line)) {
      flushParagraph();
      flushTable();
      flushList();
      blocks.push(`<div class="eaa-math-block">${escapeHtml(line.trim())}</div>`);
      continue;
    }
    if (list && /^\s+/.test(line) && list.items.length) {
      const itemIndent = list.itemIndents[list.itemIndents.length - 1] ?? 0;
      const prefix = " ".repeat(itemIndent);
      const content = line.startsWith(prefix) ? line.slice(itemIndent) : line.trimStart();
      list.items[list.items.length - 1] += `\n${content}`;
      continue;
    }
    const unorderedItem = line.match(/^([-*+]\s+)(.+)$/);
    const orderedItem = line.match(/^((\d+)[.)]\s+)(.+)$/);
    if (unorderedItem || orderedItem) {
      flushParagraph();
      flushTable();
      const ordered = Boolean(orderedItem);
      if (!list || list.ordered !== ordered) flushList();
      if (!list) list = { ordered, items: [], itemIndents: [], start: orderedItem ? Number(orderedItem[2]) : null };
      list.items.push(unorderedItem?.[2] ?? orderedItem?.[3] ?? "");
      list.itemIndents.push(unorderedItem?.[1].length ?? orderedItem?.[1].length ?? 0);
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
      if (list && list.items.length) {
        list.items[list.items.length - 1] += "\n";
        continue;
      }
      flushParagraph();
      flushList();
      continue;
    }
    flushList();
    paragraph.push(line);
  }
  if (displayMath !== null) blocks.push(`<div class="eaa-math-block">${escapeHtml(displayMath.lines.join("\n"))}</div>`);
  if (code !== null) blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushTable();
  flushList();
  flushParagraph();
  return blocks.join("\n");
};

export { escapeHtml };
