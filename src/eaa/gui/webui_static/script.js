/* globals marked */
(function () {
  const messagesEl = document.getElementById("messages");
  const imagesSidebarEl = document.getElementById("imagesSidebar");
  const inputEl = document.getElementById("inputBox");
  const sendBtn = document.getElementById("sendBtn");
  const fileInput = document.getElementById("fileInput");
  const attachBtn = document.getElementById("attachBtn");
  const statusEl = document.getElementById("status");

  let lastMessageId = null;
  let polling = true;
  let userPinnedScroll = false;

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  // Custom markdown renderer with precise control
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
    
    // Step 3: Process headers (# ## ### etc.)
    raw = raw.replace(/^(#{1,6})\s+(.+)$/gm, function (_m, hashes, text) {
      const level = hashes.length;
      return `<h${level}>${text.trim()}</h${level}>`;
    });
    
    // Step 4: Process bold (**text**) and italic (*text*)
    // Bold first to avoid conflicts
    raw = raw.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    raw = raw.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    
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
    
    // Step 6: Escape remaining HTML and convert newlines to <br>
    // Split by existing HTML tags to avoid double-escaping
    let parts = raw.split(/(<[^>]+>)/);
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) { // Text parts (not HTML tags)
        parts[i] = escapeHtml(parts[i]).replace(/\n/g, '<br>');
      }
    }
    raw = parts.join('');
    
    // Step 7: Restore code blocks and inline codes
    raw = raw.replace(/@@CODEBLOCK_(\d+)@@/g, function (_m, i) {
      return codeBlocks[Number(i)] || "";
    });
    raw = raw.replace(/@@INLINECODE_(\d+)@@/g, function (_m, i) {
      return inlineCodes[Number(i)] || "";
    });
    
    return raw;
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
    timestamp.textContent = msg.timestamp;
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

    if (msg.role === "system") {
      content.innerHTML = escapeHtml(fullContent).replace(/\n/g, '<br>');
    } else {
      content.innerHTML = renderMarkdown(fullContent);
    }

    container.appendChild(meta);
    container.appendChild(content);

    // Inline image from base64 column
    if (msg.image) {
      const img = document.createElement("img");
      img.className = "inline";
      img.src = msg.image;
      container.appendChild(img);
      addImageToSidebar(msg.image);
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
      container.appendChild(img);
      addImageToSidebar(url);
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

  async function handleClipboardPaste(event) {
    const items = event.clipboardData?.items;
    if (!items) return;

    for (let item of items) {
      if (item.type.startsWith('image/')) {
        event.preventDefault();
        
        const file = item.getAsFile();
        if (!file) continue;

        try {
          // Show loading state
          const originalPlaceholder = inputEl.placeholder;
          inputEl.placeholder = "Uploading image...";
          inputEl.disabled = true;

          // Convert to base64
          const reader = new FileReader();
          const imageData = await new Promise((resolve, reject) => {
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });

          // Upload to server
          const filePath = await uploadClipboardImage(imageData);

          // Insert image tag into input
          const current = inputEl.value;
          const toInsert = current && !/\s$/.test(current) ? ` <img ${filePath}>` : `<img ${filePath}>`;
          inputEl.value = current + toInsert;

          // Restore input state
          inputEl.placeholder = originalPlaceholder;
          inputEl.disabled = false;
          inputEl.focus();

        } catch (e) {
          alert(`Failed to paste image: ${e.message}`);
          inputEl.placeholder = originalPlaceholder;
          inputEl.disabled = false;
        }
        
        break; // Only handle the first image
      }
    }
  }

  async function sendMessage() {
    const content = inputEl.value.trim();
    if (!content) return;
    sendBtn.disabled = true;
    try {
      const res = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
      // Show loading state
      const originalText = attachBtn.textContent;
      attachBtn.textContent = "📤";
      attachBtn.disabled = true;
      
      // Convert file to base64
      const reader = new FileReader();
      const imageData = await new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      // Upload to server
      const filePath = await uploadClipboardImage(imageData);

      // Insert image tag into input
      const current = inputEl.value;
      const toInsert = current && !/\s$/.test(current) ? ` <img ${filePath}>` : `<img ${filePath}>`;
      inputEl.value = current + toInsert;
      inputEl.focus();

      // Restore button state
      attachBtn.textContent = originalText;
      attachBtn.disabled = false;

    } catch (e) {
      alert(`Failed to upload file: ${e.message}`);
      attachBtn.textContent = "📎";
      attachBtn.disabled = false;
    }
    
    // Clear the file input for next use
    fileInput.value = "";
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
