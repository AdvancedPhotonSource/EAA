// /public/autoscroll.js
(function () {
  console.log("✅ sticky autoscroll.js (large threshold + image-proof)");

  // Bump this to be more forgiving (e.g., 600–1000)
  const STICKY_THRESHOLD = (window.AUTOSCROLL_THRESHOLD ?? 200);

  // Message-ish selectors to detect the main chat pane (avoid sidebar)
  const MESSAGE_SELECTORS = [
    '[data-testid="messages"]',
    '[data-testid*="message"]',
    '[id^="message-"]',
    'main [class*="message"]',
    'main [data-radix-scroll-area-viewport]'
  ];

  function isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < STICKY_THRESHOLD;
  }

  function allScrollables() {
    const nodes = Array.from(document.querySelectorAll('*'));
    return nodes.filter((el) => {
      const cs = getComputedStyle(el);
      const canScroll = /(auto|scroll)/.test(cs.overflowY);
      const overflown = el.scrollHeight > el.clientHeight + 1;
      return canScroll && overflown;
    });
  }

  function containsAnySelector(root, sels) {
    return sels.some(sel => root.querySelector(sel));
  }

  function chatScrollables() {
    const candidates = allScrollables().filter(el => containsAnySelector(el, MESSAGE_SELECTORS));
    return candidates.length ? candidates : allScrollables();
  }

  function lastMessageEl() {
    const sels = [
      '[data-testid="messages"] > *',
      '[data-testid*="message"]',
      '[id^="message-"]',
      'main [class*="message"]',
      'main [data-radix-scroll-area-viewport] > div > *'
    ];
    for (const sel of sels) {
      const els = document.querySelectorAll(sel);
      if (els.length) return els[els.length - 1];
    }
    return null;
  }

  // — Sticky state: true if window or any chat scroller is near bottom —
  function calcSticky() {
    const doc = document.scrollingElement || document.documentElement;
    const scrollers = [doc, ...chatScrollables()];
    return scrollers.some(isNearBottom);
  }
  let sticky = true;

  function scrollAllToBottom() {
    const doc = document.scrollingElement || document.documentElement;
    doc.scrollTop = doc.scrollHeight;

    for (const el of chatScrollables()) {
      el.scrollTop = el.scrollHeight;
    }

    const last = lastMessageEl();
    if (last) last.scrollIntoView({ block: 'end' });
  }

  // — Settle loop: gently keep bottom while heights change (images/markdown) —
  let settling = false;
  function scheduleSettle(force = false) {
    if (!force && !calcSticky()) return; // respect user reading up-thread
    if (settling) return;
    settling = true;

    const MAX_MS = 1400;
    const START = performance.now();
    const targets = new Set([document.scrollingElement || document.documentElement, ...chatScrollables()]);
    const lastHeights = new Map();
    let stableFrames = 0;

    (function tick() {
      // Re-check stickiness each frame; abort if user scrolled up
      if (!force && !calcSticky()) {
        settling = false;
        return;
      }

      scrollAllToBottom();

      let changed = false;
      for (const el of targets) {
        const h = el.scrollHeight;
        if (lastHeights.get(el) !== h) {
          lastHeights.set(el, h);
          changed = true;
        }
      }
      stableFrames = changed ? 0 : (stableFrames + 1);

      const timeUp = performance.now() - START > MAX_MS;
      if (stableFrames >= 4 || timeUp) {
        scrollAllToBottom(); // final nudge
        settling = false;
        return;
      }
      requestAnimationFrame(tick);
    })();
  }

  // — Image-aware: nudge when images finish loading/decoding —
  function handleImage(img) {
    const nudge = () => scheduleSettle(false);
    if (typeof img.decode === 'function') {
      img.decode().then(nudge).catch(nudge);
    }
    if (img.complete) {
      nudge();
    } else {
      img.addEventListener('load', nudge, { once: true });
      img.addEventListener('error', nudge, { once: true });
    }
  }
  function watchImagesIn(node) {
    if (!node) return;
    if (node.tagName === 'IMG') handleImage(node);
    const imgs = node.querySelectorAll ? node.querySelectorAll('img') : [];
    imgs.forEach(handleImage);
  }

  // — Observe DOM & size changes —
  const mo = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === 'childList') m.addedNodes && m.addedNodes.forEach(watchImagesIn);
    }
    scheduleSettle(false);
  });

  const ro = new ResizeObserver(() => scheduleSettle(false));

  function start() {
    mo.observe(document.body, { childList: true, subtree: true });
    ro.observe(document.body);
    chatScrollables().forEach(el => ro.observe(el));
    document.querySelectorAll('img').forEach(handleImage);

    // Keep stickiness up to date as the user scrolls
    const targets = new Set([window, document, document.scrollingElement || document.documentElement, ...chatScrollables()]);
    for (const t of targets) {
      (t === window ? window : t).addEventListener('scroll', () => { sticky = calcSticky(); }, { passive: true });
    }

    // Safety tick
    setInterval(() => { sticky = calcSticky(); }, 1000);

    // Initial settle (respects stickiness)
    sticky = true; // start pinned on first load
    scheduleSettle(false);
  }

  // Optional: server ping from Python after .send()
  window.addEventListener('message', (e) => {
    if (e?.data === 'scrollBottom') scheduleSettle(false);
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
