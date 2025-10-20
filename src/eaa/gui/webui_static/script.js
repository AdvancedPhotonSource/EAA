/* globals marked */
(function () {
  const messagesEl = document.getElementById("messages");
  const imagesSidebarEl = document.getElementById("imagesSidebar");
  const inputEl = document.getElementById("inputBox");
  const sendBtn = document.getElementById("sendBtn");
  const fileInput = document.getElementById("fileInput");
  const attachBtn = document.getElementById("attachBtn");
  const statusEl = document.getElementById("status");
  const inputRow = document.getElementById("inputRow");
  const imageModal = document.getElementById("imageModal");
  const imageModalImg = document.getElementById("imageModalImg");
  const imageModalClose = document.getElementById("imageModalClose");

  let lastMessageId = null;
  let polling = true;
  let userPinnedScroll = false;
  let dragCounter = 0;

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  // Custom markdown renderer with precise control
  //
  // Workflow overview:
  //   1. Temporarily replace markdown constructs (code blocks, inline code,
  //      headers, bold, italics) with placeholders so we can treat the rest
  //      of the message as raw text.
  //   2. Normalize whitespace and list markers.
  //   3. HTML-escape the remaining text wholesale, turning user-provided
  //      angle brackets into entities (e.g. `<y>` -> `&lt;y&gt;`).
  //   4. Rehydrate the placeholders into safe HTML snippets.
  //
  // This placeholder approach keeps markdown styling while ensuring literal
  // angle brackets are displayed instead of being interpreted as tags.
  function renderMarkdown(text) {
    if (!text) return "";
    
    let raw = String(text);
    
    // Step 1: Extract and protect code blocks (```...```)
    const codeBlocks = [];
    raw = raw.replace(/```([\s\S]*?)```/g, function (_m, code) {
      const idx = codeBlocks.length;
      codeBlocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
      return `@@CODEBLOCK_${idx}@@`;
    });
    
    // Step 2: Extract and protect inline code (`...`)
    const inlineCodes = [];
    raw = raw.replace(/`([^`\n]+)`/g, function (_m, code) {
      const idx = inlineCodes.length;
      inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
      return `@@INLINECODE_${idx}@@`;
    });
    
    // Step 3: Process headers (# ## ### etc.) using placeholders
    const headers = [];
    raw = raw.replace(/^(#{1,6})\s+(.+)$/gm, function (_m, hashes, headerText) {
      const idx = headers.length;
      headers.push({ level: hashes.length, text: headerText.trim() });
      return `@@HEADER_${idx}@@`;
    });
    
    // Step 4: Process bold (**text**) and italic (*text*) using placeholders
    const bolds = [];
    raw = raw.replace(/\*\*([^*\n]+)\*\*/g, function (_m, boldText) {
      const idx = bolds.length;
      bolds.push(boldText);
      return `@@BOLD_${idx}@@`;
    });
    
    const italics = [];
    raw = raw.replace(/\*([^*\n]+)\*/g, function (_m, italicText) {
      const idx = italics.length;
      italics.push(italicText);
      return `@@ITALIC_${idx}@@`;
    });
    
    // Step 5: Aggressively remove leading whitespace and list formatting
    // Split into lines and process each one
    let lines = raw.split('\n');
    let processedLines = [];
    
    for (let line of lines) {
      // Remove excessive leading whitespace (keep max 2 spaces for basic indentation)
      let trimmed = line.replace(/^[ \t]+/, '');
      
      // Remove list markers entirely at the start of lines
      trimmed = trimmed.replace(/^(\d+)\.\s*/, '$1. ');
      trimmed = trimmed.replace(/^[-*+]\s*/, '- ');
      
      processedLines.push(trimmed);
    }
    
    raw = processedLines.join('\n');
    
    // Step 6: Escape all HTML characters then convert newlines to <br>
    raw = escapeHtml(raw).replace(/\n/g, '<br>');
    
    // Step 7: Restore placeholders and protected code blocks
    raw = raw.replace(/@@HEADER_(\d+)@@/g, function (_m, i) {
      const header = headers[Number(i)];
      if (!header) return "";
      const level = Math.min(Math.max(header.level, 1), 6);
      return `<h${level}>${escapeHtml(header.text)}</h${level}>`;
    });

    raw = raw.replace(/@@BOLD_(\d+)@@/g, function (_m, i) {
      const text = bolds[Number(i)];
      if (text == null) return "";
      return `<strong>${escapeHtml(text)}</strong>`;
    });

    raw = raw.replace(/@@ITALIC_(\d+)@@/g, function (_m, i) {
      const text = italics[Number(i)];
      if (text == null) return "";
      return `<em>${escapeHtml(text)}</em>`;
    });

    raw = raw.replace(/@@CODEBLOCK_(\d+)@@/g, function (_m, i) {
      return codeBlocks[Number(i)] || "";
    });
    raw = raw.replace(/@@INLINECODE_(\d+)@@/g, function (_m, i) {
      return inlineCodes[Number(i)] || "";
    });
    
    return raw;
  }

  function formatToolApprovalMessage(text) {
    const original = text == null ? '' : String(text);
    const lines = original.split('\n');
    const placeholder = '<see formatted code below>';
    let codeSnippet = null;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const match = line.match(/^(\s*Arguments:\s*)(\{.*\})\s*$/);
      if (!match) {
        continue;
      }

      let jsonPart = match[2];
      let parsed;
      try {
        parsed = JSON.parse(jsonPart);
      } catch (_err) {
        continue;
      }

      let replaced = false;
      if (typeof parsed.code === 'string') {
        codeSnippet = parsed.code;
        parsed.code = placeholder;
        replaced = true;
      } else if (parsed.arguments && typeof parsed.arguments.code === 'string') {
        codeSnippet = parsed.arguments.code;
        parsed.arguments = { ...parsed.arguments, code: placeholder };
        replaced = true;
      }
      if (!replaced) {
        continue;
      }
      
      const prefix = match[1];
      lines[i] = `${prefix}${JSON.stringify(parsed)}`;
      break;
    }

    if (!codeSnippet) {
      return { text: original, code: null };
    }

    const normalisedCode = codeSnippet.replace(/\r\n/g, '\n');
    return { text: lines.join('\n'), code: normalisedCode };
  }

  function formatTimestamp(raw) {
    if (!raw) return "";
    const str = String(raw).trim();
    const digits = str.replace(/\D/g, "");
    if (digits.length < 14) {
      return str;
    }

    const year = digits.slice(0, 4);
    const month = digits.slice(4, 6);
    const day = digits.slice(6, 8);
    const hour = digits.slice(8, 10);
    const minute = digits.slice(10, 12);
    const second = digits.slice(12, 14);
    let formatted = `${year}-${month}-${day} ${hour}:${minute}:${second}`;

    if (digits.length > 14) {
      const micro = digits.slice(14, 20).padEnd(6, "0");
      formatted += `.${micro}`;
    }

    return formatted;
  }

  function isNearBottom(container = messagesEl) {
    const threshold = 50;
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    return distanceFromBottom <= threshold;
  }

  function scrollToBottom(container = messagesEl, behavior = "smooth") {
    requestAnimationFrame(() => {
      container.scrollTo({ 
        top: container.scrollHeight, 
        behavior: behavior 
      });
    });
  }

  function parseImageTagFromContent(content) {
    const m = (content || "").match(/<img\s+([^>\s]+)>/);
    if (m && m[1]) {
      return m[1];
    }
    return null;
  }

  function getRoleDisplayName(role) {
    const roleMap = {
      'user': 'user',
      'assistant': 'assistant', 
      'tool': 'tool',
      'system': 'system',
      'user_webui': 'user_webui'
    };
    return roleMap[role] || role;
  }

  function createMessageElement(msg) {
    const container = document.createElement("div");
    container.className = "message";

    const meta = document.createElement("div");
    meta.className = "meta";

    const roleTag = document.createElement("span");
    roleTag.className = `role-tag ${getRoleDisplayName(msg.role)}`;
    roleTag.textContent = msg.role;

    const timestamp = document.createElement("span");
    timestamp.textContent = formatTimestamp(msg.timestamp);
    timestamp.style.color = "var(--muted)";

    meta.appendChild(roleTag);
    meta.appendChild(timestamp);

    const content = document.createElement("div");
    content.className = "content";
    
    // Build the full content including tool calls
    let fullContent = String(msg.content || "");

    if (msg.tool_calls) {
      if (fullContent) {
        fullContent += "\n\n";
      }
      fullContent += `**[Tool calls]**\n${msg.tool_calls}`;
    }

    const formattedApproval = formatToolApprovalMessage(fullContent);
    fullContent = formattedApproval.text;

    const normalisedContent = fullContent.replace(/\r\n/g, "\n");
    const normalisedLines = normalisedContent ? normalisedContent.split("\n") : [];
    const lineCount = normalisedLines.length;
    const isUserRole = msg.role === "user" || msg.role === "user_webui";
    const shouldFold = isUserRole && lineCount > 10;
    let actionsRow = null;

    if (shouldFold) {
      content.classList.add("foldable");
      const previewLines = normalisedLines.slice(0, 10);
      previewLines.push("…");
      const collapsedText = previewLines.join("\n");
      const collapsedHtml = renderMarkdown(collapsedText);
      const expandedHtml = renderMarkdown(fullContent);
      let expanded = false;

      const renderState = (expand) => {
        content.innerHTML = expand ? expandedHtml : collapsedHtml;
        content.classList.toggle("folded", !expand);
        if (expand && formattedApproval.code) {
          const pre = document.createElement("pre");
          pre.className = "approval-code-block";
          const codeEl = document.createElement("code");
          codeEl.textContent = formattedApproval.code;
          pre.appendChild(codeEl);
          content.appendChild(pre);
        }
      };

      renderState(false);

      const toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.className = "show-more-button";
      toggleBtn.textContent = "Show more";
      toggleBtn.addEventListener("click", () => {
        expanded = !expanded;
        renderState(expanded);
        toggleBtn.textContent = expanded ? "Show less" : "Show more";
      });

      actionsRow = document.createElement("div");
      actionsRow.className = "message-actions";
      actionsRow.appendChild(toggleBtn);
    } else {
      content.innerHTML = renderMarkdown(fullContent);
      content.classList.remove("foldable");
      content.classList.remove("folded");

      if (formattedApproval.code) {
        const pre = document.createElement("pre");
        pre.className = "approval-code-block";
        const codeEl = document.createElement("code");
        codeEl.textContent = formattedApproval.code;
        pre.appendChild(codeEl);
        content.appendChild(pre);
      }
    }

    container.appendChild(meta);
    container.appendChild(content);
    maybeAddApprovalActions(content, msg);

    if (actionsRow) {
      container.appendChild(actionsRow);
    }

    // Inline image from base64 column
    if (msg.image) {
      const img = document.createElement("img");
      img.className = "inline";
      img.src = msg.image;
      img.dataset.fullSrc = msg.image;
      container.appendChild(img);
      addImageToSidebar(msg.image);
      setupImagePreview(img);
    }

    // Inline image from <img path>
    const shouldParseContentImage = msg.role !== "system";
    const pathInText = shouldParseContentImage ? parseImageTagFromContent(msg.content || "") : null;
    if (pathInText) {
      const url = `/api/image?path=${encodeURIComponent(pathInText)}`;
      const img = document.createElement("img");
      img.className = "inline";
      img.loading = "lazy";
      img.src = url;
      img.dataset.fullSrc = url;
      container.appendChild(img);
      addImageToSidebar(url);
      setupImagePreview(img);
    }

    return container;
  }

  function addImageToSidebar(src) {
    const box = document.createElement("div");
    box.className = "thumb";
    const img = document.createElement("img");
    img.src = src;
    img.loading = "lazy";
    box.appendChild(img);
    
    imagesSidebarEl.appendChild(box);
    
    setTimeout(() => {
      scrollToBottom(imagesSidebarEl);
    }, 100);
  }

  async function fetchMessages() {
    const qs = lastMessageId ? `?since_id=${encodeURIComponent(lastMessageId)}` : "";
    const res = await fetch(`/api/messages${qs}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.messages || [];
  }

  function renderMessages(newMessages) {
    if (!newMessages.length) return;
    
    const shouldAutoScroll = !userPinnedScroll || isNearBottom();

    for (const msg of newMessages) {
      const el = createMessageElement(msg);
      messagesEl.appendChild(el);
      lastMessageId = msg.id;
    }

    if (shouldAutoScroll) {
      setTimeout(() => {
        scrollToBottom(messagesEl);
      }, 150);
    }
  }

  async function poll() {
    while (polling) {
      try {
        const messages = await fetchMessages();
        if (messages.length > 0) {
          renderMessages(messages);
        }
        statusEl.textContent = "Connected";
      } catch (e) {
        statusEl.textContent = "Reconnecting...";
      }
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  async function uploadClipboardImage(imageData) {
    try {
      const res = await fetch("/api/upload-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: imageData })
      });
      
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || `HTTP ${res.status}`);
      }
      
      const result = await res.json();
      return result.file_path;
    } catch (e) {
      console.error("Failed to upload image:", e);
      throw e;
    }
  }

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not read file data"));
      reader.readAsDataURL(file);
    });
  }

  function appendImageTag(filePath) {
    const current = inputEl.value;
    const needsSpace = current && !/\s$/.test(current);
    const toInsert = needsSpace ? ` <img ${filePath}>` : `<img ${filePath}>`;
    inputEl.value = current + toInsert;
  }

  function openImageModal(src) {
    if (!imageModal || !imageModalImg) return;
    imageModalImg.src = src;
    imageModal.classList.add("open");
    imageModal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function closeImageModal() {
    if (!imageModal || !imageModalImg) return;
    imageModal.classList.remove("open");
    imageModal.setAttribute("aria-hidden", "true");
    imageModalImg.src = "";
    document.body.style.overflow = "";
  }

  function setupImagePreview(img) {
    if (!img) return;
    img.addEventListener("click", () => {
      const src = img.getAttribute("data-full-src") || img.src;
      if (!src) return;
      openImageModal(src);
    });
  }

  function maybeAddApprovalActions(contentEl, msg) {
    if (!contentEl || !msg) return;
    if (msg.role !== "system") return;
    const text = String(msg.content || "");
    if (!/Approve\?\s*\[y\/N\]:/i.test(text)) return;
    if (contentEl.querySelector(".approval-actions")) return;

    const actions = document.createElement("span");
    actions.className = "approval-actions";

    const createButton = (label, value) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "approval-button";
      btn.textContent = label;
      btn.dataset.value = value;
      return btn;
    };

    const yesBtn = createButton("Yes", "yes");
    const noBtn = createButton("No", "no");
    actions.appendChild(yesBtn);
    actions.appendChild(noBtn);

    contentEl.appendChild(actions);

    let submitting = false;
    async function handleClick(event) {
      if (submitting) return;
      submitting = true;
      yesBtn.disabled = true;
      noBtn.disabled = true;
      const value = event.currentTarget.dataset.value || "";
      try {
        await postUserMessage(value);
      } catch (err) {
        const message = err && err.message ? err.message : "Failed to submit response.";
        alert(message);
        submitting = false;
        yesBtn.disabled = false;
        noBtn.disabled = false;
      }
    }

    yesBtn.addEventListener("click", handleClick);
    noBtn.addEventListener("click", handleClick);
  }

  function withInputLoading(message = "Uploading image...") {
    const originalPlaceholder = inputEl.placeholder;
    const wasDisabled = inputEl.disabled;
    inputEl.placeholder = message;
    inputEl.disabled = true;
    return () => {
      inputEl.placeholder = originalPlaceholder;
      inputEl.disabled = wasDisabled;
    };
  }

  function withAttachBusy(tempText = "📤") {
    if (!attachBtn) {
      return () => {};
    }
    const originalText = attachBtn.textContent;
    const wasDisabled = attachBtn.disabled;
    attachBtn.textContent = tempText;
    attachBtn.disabled = true;
    return () => {
      attachBtn.textContent = originalText;
      attachBtn.disabled = wasDisabled;
    };
  }

  function isImageFile(file) {
    if (!file) return false;
    if (file.type) {
      return file.type.startsWith("image/");
    }
    return /\.(png|jpe?g|gif|webp|bmp|tiff?)$/i.test(file.name || "");
  }

  async function attachImageFromFile(file, { onStart } = {}) {
    if (!isImageFile(file)) {
      throw new Error("Only image files are supported");
    }

    let cleanup = () => {};
    if (typeof onStart === "function") {
      const maybeCleanup = onStart();
      if (typeof maybeCleanup === "function") {
        cleanup = maybeCleanup;
      }
    }

    try {
      const imageData = await readFileAsDataURL(file);
      const filePath = await uploadClipboardImage(imageData);
      appendImageTag(filePath);
    } finally {
      cleanup();
    }
  }

  async function handleClipboardPaste(event) {
    const items = event.clipboardData?.items;
    if (!items) return;

    for (let item of items) {
      if (item.type.startsWith('image/')) {
        event.preventDefault();
        
        const file = item.getAsFile();
        if (!file) continue;

        try {
          await attachImageFromFile(file, {
            onStart: () => withInputLoading("Uploading image...")
          });
          inputEl.focus();
        } catch (e) {
          const message = e?.message || "Failed to paste image.";
          alert(`Failed to paste image: ${message}`);
        }
        
        break; // Only handle the first image
      }
    }
  }

  async function postUserMessage(content) {
    const res = await fetch("/api/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content })
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      const message = error.error || `HTTP ${res.status}`;
      throw new Error(message);
    }
    return res.json().catch(() => ({}));
  }

  async function sendMessage() {
    const content = inputEl.value.trim();
    if (!content) return;
    sendBtn.disabled = true;
    try {
      await postUserMessage(content);
      inputEl.value = "";
      inputEl.focus();
      
      setTimeout(() => {
        scrollToBottom(messagesEl);
        userPinnedScroll = false;
      }, 100);
    } catch (e) {
      alert("Failed to send message: " + e.message);
    } finally {
      sendBtn.disabled = false;
    }
  }

  // Attach button opens file browser
  attachBtn.addEventListener("click", () => {
    fileInput.click();
  });

  // File input change handler for attach button
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    
    try {
      await attachImageFromFile(file, {
        onStart: () => {
          const cleanups = [
            withInputLoading("Uploading image..."),
            withAttachBusy("📤")
          ];
          return () => cleanups.forEach(fn => fn());
        }
      });
      inputEl.focus();
    } catch (e) {
      const message = e?.message || "Failed to upload file.";
      alert(`Failed to upload file: ${message}`);
    }
    
    // Clear the file input for next use
    fileInput.value = "";
  });

  function isFileDrag(event) {
    const dt = event.dataTransfer;
    if (!dt) return false;
    if (dt.types && typeof dt.types.includes === "function") {
      return dt.types.includes("Files");
    }
    if (Array.isArray(dt.types)) {
      return dt.types.includes("Files");
    }
    return dt.files && dt.files.length > 0;
  }

  function firstImageFileFromEvent(event) {
    const dt = event.dataTransfer;
    if (!dt || !dt.files || !dt.files.length) {
      return null;
    }
    for (const file of dt.files) {
      if (isImageFile(file)) {
        return file;
      }
    }
    return null;
  }

  if (inputRow) {
    inputRow.addEventListener("dragenter", (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragCounter += 1;
      inputRow.classList.add("drag-active");
    });

    inputRow.addEventListener("dragover", (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "copy";
      }
    });

    inputRow.addEventListener("dragleave", (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragCounter = Math.max(0, dragCounter - 1);
      if (dragCounter === 0) {
        inputRow.classList.remove("drag-active");
      }
    });

    inputRow.addEventListener("drop", async (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragCounter = 0;
      inputRow.classList.remove("drag-active");

      const file = firstImageFileFromEvent(event);
      if (!file) {
        alert("Only image files can be attached.");
        return;
      }

      try {
        await attachImageFromFile(file, {
          onStart: () => withInputLoading("Uploading image...")
        });
        inputEl.focus();
      } catch (e) {
        const message = e?.message || "Failed to upload file.";
        alert(`Failed to upload file: ${message}`);
      } finally {
        if (event.dataTransfer) {
          event.dataTransfer.clearData();
        }
      }
    });
  }

  if (imageModalClose) {
    imageModalClose.addEventListener("click", () => {
      closeImageModal();
    });
  }

  if (imageModal) {
    imageModal.addEventListener("click", (event) => {
      if (event.target === imageModal) {
        closeImageModal();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && imageModal && imageModal.classList.contains("open")) {
      closeImageModal();
    }
  });

  // Clipboard paste event listener
  inputEl.addEventListener("paste", handleClipboardPaste);

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  sendBtn.addEventListener("click", sendMessage);

  messagesEl.addEventListener("scroll", () => {
    userPinnedScroll = !isNearBottom();
  });

  inputEl.addEventListener("focus", () => {
    if (isNearBottom()) {
      userPinnedScroll = false;
    }
  });

  // Initialize
  setTimeout(() => {
    scrollToBottom(messagesEl, "auto");
  }, 500);
  
  poll();
})(); 
