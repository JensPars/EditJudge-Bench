// Interactive edits figure: example cards float out of a tile when the cursor is
// near it, and a click docks them into a card with the judges' scores.
(() => {
  "use strict";
  const stage = document.querySelector(".ex-stage");
  if (!stage) return;

  // Figure units = the SVG viewBox (Figma frame scaled by 1025 / 2590.5).
  const FW = 1025, FH = 759, S = 1025 / 2590.5;
  const tile = (key, label, fx, fy) =>
    ({ key, label, x: (fx + 0.5) * S, y: fy * S, w: 643.5 * S, h: 475 * S });
  const TILES = [
    tile("lighting", "Lighting", -0.5, 0),
    tile("removal", "Remove / add", 648.5, 1),
    tile("scale", "Scale", 1297.5, 1),
    tile("material", "Material", 1946.5, 0),
    tile("movement", "Move", -0.5, 479),
    tile("camera", "Camera", 1946.5, 483),
    tile("rotation", "Rotate", -0.5, 958),
    tile("articulation", "Articulate", -0.5, 1437),
    tile("shape", "Shape", 648.5, 1437),
  ];
  const NEAR = 28;                       // neighbourhood that wakes a tile up
  const PEEK = { w: 312, h: 181, r: 12 }; // front card: 300x169 image + 6px white border
  const CARD_W = FW - 16;                 // the card spans the figure
  const SHEET_BELOW = 860;                // narrower stages get a bottom sheet instead
  const CENTER = { x: 530, y: 374 };      // the 3D scene in the middle
  const KIND = {
    positive: "Correct instruction",
    counterfactual: "Directional counterfactual",
    cross_type: "Cross-edit-type counterfactual",
    wrong_object: "Wrong-object counterfactual",
    no_edit: "No-edit counterfactual",
  };
  const IMG = (id, side) => `assets/examples/${id}_${side}.webp`;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const NS = "http://www.w3.org/2000/svg";

  // ---------- springs ----------
  class Sp {
    constructor(x = 0) { this.x = x; this.v = 0; this.t = x; }
    set(x) { this.x = this.t = x; this.v = 0; }
    step(dt, k, c) {
      if (reduce.matches) { this.x = this.t; this.v = 0; return; }
      this.v += (k * (this.t - this.x) - c * this.v) * dt;
      this.x += this.v * dt;
    }
    rest(eps) { return Math.abs(this.t - this.x) < eps && Math.abs(this.v) < eps * 8; }
  }
  const V2 = () => ({ x: new Sp(), y: new Sp() });
  const R4 = () => ({ x: new Sp(), y: new Sp(), w: new Sp(), h: new Sp() });
  const each = (o) => Object.values(o);
  const to = (o, v) => { for (const k in v) o[k].t = v[k]; };
  const snap = (o, v) => { for (const k in v) o[k].set(v[k]); };
  const stepAll = (o, dt, k, c) => each(o).forEach((s) => s.step(dt, k, c));
  const restAll = (o, eps) => each(o).every((s) => s.rest(eps));

  // ---------- DOM ----------
  const h = (tag, cls, attrs) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };
  const s = (tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };
  const box = { width: FW, height: FH, viewBox: `0 0 ${FW} ${FH}` };

  const fx = h("div", "ex-fx");
  const under = s("svg", { ...box, class: "ex-under", "aria-hidden": "true" });
  const veil = s("path", { class: "ex-veil", "fill-rule": "evenodd" });
  const hovRing = s("rect", { class: "ex-ring", rx: 4 });
  const selRing = s("rect", { class: "ex-ring ex-sel", rx: 4 });
  const pulses = TILES.map((t, i) => s("rect", {
    class: "ex-pulse", x: t.x + 1, y: t.y + 1, width: t.w - 2, height: t.h - 2, rx: 4,
    style: `animation-delay:${i * 70}ms`,
  }));
  under.append(veil, ...pulses, hovRing, selRing);

  const backs = [0, 1].map(() => {
    const d = h("div", "ex-back");
    d.append(h("img", "", { alt: "" }));
    fx.append(d);
    return d;
  });

  const bub = s("svg", { ...box, class: "ex-bub", "aria-hidden": "true" });
  const defs = s("defs");
  const filt = s("filter", { id: "exShadow", x: "-40%", y: "-40%", width: "180%", height: "180%" });
  filt.append(s("feDropShadow", { dx: 0, dy: 8, stdDeviation: 10, "flood-color": "#15294b", "flood-opacity": 0.24 }));
  defs.append(filt);
  const gPeek = s("g", { filter: "url(#exShadow)" });
  const peekTail = s("path", { class: "ex-paper" });
  const peekRect = s("rect", { class: "ex-paper", x: -PEEK.w / 2, y: -PEEK.h / 2, width: PEEK.w, height: PEEK.h, rx: PEEK.r });
  gPeek.append(peekTail, peekRect);
  const gCard = s("g", { filter: "url(#exShadow)" });
  const cardTail = s("path", { class: "ex-paper" });
  const cardRect = s("rect", { class: "ex-paper", rx: 14 });
  gCard.append(cardTail, cardRect);
  bub.append(defs, gCard, gPeek);

  const front = h("div", "ex-front");
  const fB = h("img", "", { alt: "" }), fA = h("img", "ex-a", { alt: "" });
  const chip = h("span", "ex-chip");
  front.append(fB, fA, chip);

  const buttons = TILES.map((t) => {
    const b = h("button", "ex-tile", { type: "button", "aria-label": `${t.label}: show examples` });
    Object.assign(b.style, { left: `${t.x}px`, top: `${t.y}px`, width: `${t.w}px`, height: `${t.h}px` });
    b.dataset.key = t.key;
    return b;
  });
  fx.prepend(under);
  fx.append(bub, front, ...buttons);

  const card = h("div", "ex-card", { role: "dialog", "aria-label": "Examples", tabindex: "-1" });
  card.innerHTML = `
    <button class="ex-close" type="button" aria-label="Close">
      <svg viewBox="0 0 12 12" width="10" height="10"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
    </button>
    <div class="ex-in">
      <div class="ex-imgs">
        <div class="ex-img"><img alt=""><span>Before</span></div>
        <div class="ex-img"><img alt=""><span>After</span></div>
      </div>
      <ul class="ex-rows"></ul>
      <div class="ex-foot">
        <div class="ex-dots" role="group" aria-label="Example"></div>
        <div class="ex-judges" role="group" aria-label="Judge"></div>
      </div>
    </div>`;
  const cIn = card.querySelector(".ex-in");
  const [cB, cA] = card.querySelectorAll(".ex-img img");
  const rowsEl = card.querySelector(".ex-rows");
  const dotsEl = card.querySelector(".ex-dots");
  const judgesEl = card.querySelector(".ex-judges");
  stage.append(fx, card);

  // ---------- state ----------
  let data = null, k = 1, sheet = false, cursor = null, mouse = true;
  const peek = { tile: null, a: new Sp(), p: V2(), b: [V2(), V2()], tip: V2(), t0: 0, after: null };
  const cs = { tile: null, pt: null, open: false, ex: 0, judge: 0, H: 400, a: new Sp(), r: R4(), tip: V2(), w0: 40 };
  const veilA = new Sp(), hole = R4();
  const hov = { tile: null, a: new Sp(), r: R4() };

  const toFU = (e) => {
    const b = stage.getBoundingClientRect();
    return { x: (e.clientX - b.left) / k, y: (e.clientY - b.top) / k };
  };
  const dist = (t, p) => Math.hypot(Math.max(t.x - p.x, 0, p.x - t.x - t.w), Math.max(t.y - p.y, 0, p.y - t.y - t.h));
  const nearest = (p) => {
    let best = null, bd = NEAR;
    for (const t of TILES) { const d = dist(t, p); if (d <= bd) { bd = d; best = t; } }
    return best;
  };
  const inside = (t, p, m) => ({ x: clamp(p.x, t.x + m, t.x + t.w - m), y: clamp(p.y, t.y + m, t.y + t.h - m) });
  const rectOf = (t) => ({ x: t.x, y: t.y, w: t.w, h: t.h });
  const exs = (t) => (data && data.types[t.key]) || [];

  // all: every image of the tile (card); otherwise just what the peek shows.
  const loaded = new Set();
  const preload = (t, all) => exs(t).forEach((e, i) => {
    if (!all && i > 2) return;
    for (const side of all || i === 0 ? ["b", "a"] : ["a"]) {
      const src = IMG(e.id, side);
      if (!loaded.has(src)) { loaded.add(src); new Image().src = src; }
    }
  });

  // ---------- peek: cards that float toward the cursor ----------
  function setPeek(t) {
    if (t === peek.tile) return;
    const fresh = peek.a.x < 0.05;
    peek.tile = t;
    if (!t) { peek.a.t = 0; return; }
    const e = exs(t);
    if (!e.length) { peek.tile = null; return; }
    preload(t, true);
    fB.src = IMG(e[0].id, "b");
    fA.src = IMG(e[0].id, "a");
    backs.forEach((d, i) => { d.firstChild.src = IMG(e[(i + 1) % e.length].id, "a"); });
    peek.t0 = performance.now();
    peek.after = null;
    if (fresh) {
      const tp = inside(t, cursor, 6);
      for (const v of [peek.tip, peek.p, ...peek.b]) snap(v, tp);
      peek.a.set(0);
    } else {
      peek.a.x = Math.min(peek.a.x, 0.55); // little re-pop when hopping between tiles
    }
    peek.a.t = 1;
  }

  function peekTarget(c) {
    // Float toward the middle of the figure, leaving a fixed gap for the tail.
    const dx = CENTER.x - c.x, dy = CENTER.y - c.y;
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n;
    const reach = Math.min(PEEK.w / 2 / (Math.abs(ux) || 1e-6), PEEK.h / 2 / (Math.abs(uy) || 1e-6));
    const D = reach + 48;
    return {
      x: clamp(c.x + ux * D, PEEK.w / 2 + 6, FW - PEEK.w / 2 - 6),
      y: clamp(c.y + uy * D, PEEK.h / 2 + 6, FH - PEEK.h / 2 - 6),
    };
  }

  // A speech-bubble tail from a (rotated) rounded rect to `tip`. It leaves the rect
  // where the line to the tip crosses it, and bends against the rect's velocity.
  function tail(cx, cy, w, h, deg, r, tip, vel, maxBase) {
    const th = (deg * Math.PI) / 180, co = Math.cos(th), si = Math.sin(th);
    const dx = tip.x - cx, dy = tip.y - cy;
    const lx = dx * co + dy * si, ly = -dx * si + dy * co;
    const hw = w / 2, hh = h / 2;
    if (Math.abs(lx) < hw - 1 && Math.abs(ly) < hh - 1) return "";
    const side = hw / (Math.abs(lx) || 1e-6) < hh / (Math.abs(ly) || 1e-6) ? "v" : "h";
    let bx, by, ux, uy, bw = maxBase;
    for (let pass = 0; pass < 2; pass++) {
      if (side === "v") {
        bx = Math.sign(lx) * (hw - 2);
        const lim = hh - r - bw / 2;
        by = lim > 0 ? clamp((ly * hw) / Math.abs(lx), -lim, lim) : 0;
        ux = 0; uy = 1;
      } else {
        by = Math.sign(ly) * (hh - 2);
        const lim = hw - r - bw / 2;
        bx = lim > 0 ? clamp((lx * hh) / Math.abs(ly), -lim, lim) : 0;
        ux = 1; uy = 0;
      }
      bw = clamp(Math.hypot(lx - bx, ly - by) * 0.42, 0, maxBase);
    }
    if (bw < 1) return "";
    const tx = lx - bx, ty = ly - by, len = Math.hypot(tx, ty);
    let px = -ty / len, py = tx / len;
    if (px * ux + py * uy < 0) { px = -px; py = -py; }
    // velocity in the local frame, only the part across the tail bends it
    const vlx = vel.x * co + vel.y * si, vly = -vel.x * si + vel.y * co;
    const bend = clamp(-(vlx * px + vly * py) * 0.045, -len * 0.35, len * 0.35);
    const mx = bx + tx * 0.52 + px * bend, my = by + ty * 0.52 + py * bend;
    const q = bw * 0.14;
    const pts = [
      [bx + ux * bw / 2, by + uy * bw / 2],
      [mx + px * q, my + py * q],
      [lx, ly],
      [mx - px * q, my - py * q],
      [bx - ux * bw / 2, by - uy * bw / 2],
    ].map(([x, y]) => `${(cx + x * co - y * si).toFixed(1)} ${(cy + x * si + y * co).toFixed(1)}`);
    return `M${pts[0]}Q${pts[1]} ${pts[2]}Q${pts[3]} ${pts[4]}Z`;
  }

  // ---------- card ----------
  const fmt = (v) => (v == null ? "–" : v % 1 ? v.toFixed(1) : String(v));
  const ICON_OK = `<svg viewBox="0 0 16 16" width="16" height="16" aria-label="correct"><circle cx="8" cy="8" r="8" fill="#3cc6a0"/><path d="M4.6 8.3l2.2 2.2 4.6-4.8" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const ICON_NO = `<svg viewBox="0 0 16 16" width="16" height="16" aria-label="wrong"><path d="M5.2 5.2l5.6 5.6M10.8 5.2l-5.6 5.6" stroke="#a3a9b5" stroke-width="1.7" stroke-linecap="round"/></svg>`;

  function renderExample(grow) {
    const e = exs(cs.tile)[cs.ex];
    cB.src = IMG(e.id, "b");
    cA.src = IMG(e.id, "a");
    rowsEl.innerHTML = "";
    for (const p of e.prompts) {
      const li = h("li", `ex-row${p.kind === "positive" ? " pos" : ""}`, { title: KIND[p.kind] || "" });
      li.innerHTML = `${p.kind === "positive" ? ICON_OK : ICON_NO}<span class="ex-txt"></span><span class="ex-bar"><i></i></span><span class="ex-num"></span>`;
      li.querySelector(".ex-txt").textContent = p.text;
      rowsEl.append(li);
    }
    dotsEl.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-current", i === cs.ex));
    if (grow) {
      rowsEl.querySelectorAll(".ex-bar i").forEach((i) => { i.style.width = "0%"; });
      void rowsEl.offsetWidth;
    }
    renderScores();
  }

  function renderScores() {
    const e = exs(cs.tile)[cs.ex], j = cs.judge, top = data.scale;
    const pos = e.prompts[0].s[j];
    rowsEl.querySelectorAll(".ex-row").forEach((li, i) => {
      const v = e.prompts[i].s[j];
      const fooled = i > 0 && v != null && pos != null && v >= pos;
      li.classList.toggle("fool", fooled);
      li.title = fooled ? `${KIND[e.prompts[i].kind]} — scored at least as high as the correct one` : KIND[e.prompts[i].kind] || "";
      li.querySelector(".ex-bar i").style.width = `${v == null ? 0 : (100 * v) / top}%`;
      li.querySelector(".ex-num").textContent = fmt(v);
    });
    judgesEl.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", i === j));
  }

  function buildCardChrome() {
    dotsEl.innerHTML = "";
    exs(cs.tile).forEach((_, i) => {
      const b = h("button", "", { type: "button", "aria-label": `Example ${i + 1}` });
      b.addEventListener("click", () => showExample(i));
      dotsEl.append(b);
    });
    card.setAttribute("aria-label", `${cs.tile.label} examples`);
  }

  function showExample(i) {
    const n = exs(cs.tile).length;
    cs.ex = (i + n) % n;
    renderExample(true);
    measure();
    kick();
  }

  function measure() {
    cs.H = cIn.offsetHeight;
    if (cs.open && !sheet) {
      const tg = cardTarget();
      to(cs.r, tg);
      to(cs.tip, cardTip(cs.tile, cs.pt, tg));
    }
  }

  // The card spans the figure, above or below its tile (whichever side has more
  // room); for the middle rows it may reach past the figure's edge.
  function cardTarget() {
    const t = cs.tile, H = cs.H, G = 9, M = 8;
    const roomAbove = t.y - G - M, roomBelow = FH - M - (t.y + t.h + G);
    const y = roomBelow >= roomAbove ? t.y + t.h + G : t.y - G - H;
    return { x: (FW - CARD_W) / 2, y, w: CARD_W, h: H };
  }

  // Bring a card that reaches past the figure fully on screen.
  function revealCard() {
    if (sheet) return;
    const r = cardTarget(), b = stage.getBoundingClientRect();
    const navH = (document.querySelector(".nav") || {}).offsetHeight || 0;
    const top = b.top + r.y * k, bottom = top + r.h * k;
    let dy = 0;
    if (bottom > innerHeight - 16) dy = bottom - innerHeight + 16;
    if (top - dy < navH + 12) dy = top - navH - 12;
    if (Math.abs(dy) > 2) scrollBy({ top: dy, behavior: reduce.matches ? "auto" : "smooth" });
  }

  // The card's tail only dips into the tile on the side facing the card, lined
  // up with where the tile was clicked; tiles off to a diagonal get the corner.
  function cardTip(t, pt, c) {
    const DIP = 16;
    const p = inside(t, pt, 24);
    const left = c.x + c.w < t.x + 20, right = c.x > t.x + t.w - 20;
    const above = c.y + c.h < t.y + 20, below = c.y > t.y + t.h - 20;
    if (left) p.x = t.x + DIP; else if (right) p.x = t.x + t.w - DIP;
    if (above) p.y = t.y + DIP; else if (below) p.y = t.y + t.h - DIP;
    return p;
  }

  function openCard(t, pt, viaKeyboard) {
    if (!data || !exs(t).length) return;
    const first = !cs.open;
    const switching = cs.tile !== t;
    cs.open = true;
    cs.tile = t;
    preload(t, true);
    if (switching) { cs.ex = 0; buildCardChrome(); renderExample(true); }
    card.classList.add("show");
    cs.H = cIn.offsetHeight;
    cs.pt = pt || { x: t.x + t.w / 2, y: t.y + t.h / 2 };
    const tip = cardTip(t, cs.pt, cardTarget());
    to(cs.tip, tip);
    if (first) {
      snap(cs.tip, tip);
      const pa = peek.a.x;
      if (!sheet && pa > 0.3) {
        const sc = 0.3 + 0.7 * pa;
        const w = PEEK.w * sc, hh = PEEK.h * sc;
        snap(cs.r, { x: peek.p.x.x - w / 2, y: peek.p.y.x - hh / 2, w, h: hh });
      } else {
        snap(cs.r, { x: tip.x - 20, y: tip.y - 15, w: 40, h: 30 });
      }
      cs.w0 = cs.r.w.x;
      if (veilA.x < 0.02) snap(hole, rectOf(t));
      cs.a.set(Math.max(cs.a.x, 0.01));
    }
    to(hole, rectOf(t));
    to(cs.r, cardTarget());
    cs.a.t = 1;
    veilA.t = 1;
    peek.tile = null; peek.a.set(0);
    hov.tile = null;
    if (viaKeyboard) card.focus({ preventScroll: true });
    revealCard();
    kick();
  }

  function closeCard() {
    if (!cs.open) return;
    cs.open = false;
    cs.a.t = 0;
    veilA.t = 0;
    const tp = { x: cs.tip.x.t, y: cs.tip.y.t };
    to(cs.r, { x: tp.x - 20, y: tp.y - 15, w: 40, h: 30 });
    cs.w0 = 40;
    card.classList.remove("show");
    const b = buttons[TILES.indexOf(cs.tile)];
    if (card.contains(document.activeElement) && b) b.focus({ preventScroll: true });
    kick();
  }

  // ---------- events ----------
  fx.addEventListener("pointermove", (e) => {
    mouse = e.pointerType !== "touch";
    cursor = toFU(e);
    const t = nearest(cursor);
    fx.classList.toggle("near", !!t);
    hov.tile = t && !(cs.open && t === cs.tile) ? t : null;
    if (!cs.open && mouse && !sheet) setPeek(t);
    kick();
  });
  fx.addEventListener("pointerleave", () => {
    cursor = null;
    hov.tile = null;
    setPeek(null);
    fx.classList.remove("near");
    kick();
  });
  fx.addEventListener("click", (e) => {
    const btn = e.target.closest(".ex-tile");
    if (e.detail === 0 && btn) {           // keyboard
      const t = TILES.find((x) => x.key === btn.dataset.key);
      return cs.open && cs.tile === t ? closeCard() : openCard(t, null, true);
    }
    const p = toFU(e);
    const t = nearest(p);
    if (t && !(cs.open && cs.tile === t)) openCard(t, p);
    else if (cs.open) closeCard();
  });
  card.querySelector(".ex-close").addEventListener("click", closeCard);
  document.addEventListener("pointerdown", (e) => {
    if (cs.open && !stage.contains(e.target)) closeCard();
  });
  document.addEventListener("keydown", (e) => {
    if (!cs.open) return;
    if (e.key === "Escape") closeCard();
    else if (e.key === "ArrowRight") showExample(cs.ex + 1);
    else if (e.key === "ArrowLeft") showExample(cs.ex - 1);
  });

  // ---------- animation loop ----------
  let raf = 0, last = 0;
  function kick() {
    if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
  }

  function step(dt) {
    // peek
    if (peek.tile && cursor) {
      to(peek.tip, inside(peek.tile, cursor, 6));
      to(peek.p, peekTarget(cursor));
    } else {
      to(peek.p, { x: peek.tip.x.x, y: peek.tip.y.x });
      peek.a.t = 0;
    }
    stepAll(peek.tip, dt, 520, 40);
    stepAll(peek.p, dt, 150, 16);
    peek.b.forEach((b, i) => {
      const o = i ? [12, -12] : [-10, -9];
      to(b, { x: peek.p.x.x + o[0], y: peek.p.y.x + o[1] });
      stepAll(b, dt, i ? 70 : 105, i ? 11 : 13);
    });
    peek.a.step(dt, 230, 19);
    // card
    stepAll(cs.r, dt, 210, 24);
    stepAll(cs.tip, dt, 260, 26);
    cs.a.step(dt, 200, 26);
    veilA.step(dt, 160, 26);
    stepAll(hole, dt, 260, 28);
    // hover ring glides between tiles
    if (hov.tile) {
      if (hov.a.x < 0.05) snap(hov.r, rectOf(hov.tile));
      to(hov.r, rectOf(hov.tile));
      hov.a.t = 1;
    } else hov.a.t = 0;
    stepAll(hov.r, dt, 420, 36);
    hov.a.step(dt, 300, 30);
  }

  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const n = Math.max(1, Math.ceil(dt / 0.008));
    for (let i = 0; i < n; i++) step(dt / n);
    render(now);
    const busy = peek.tile || !peek.a.rest(0.002) || !restAll(peek.p, 0.05) || !restAll(cs.r, 0.05) ||
      !cs.a.rest(0.002) || !veilA.rest(0.002) || !restAll(hole, 0.05) || !hov.a.rest(0.002) ||
      !restAll(hov.r, 0.05) || !restAll(cs.tip, 0.05);
    if (busy) raf = requestAnimationFrame(frame);
  }

  const vis = (el, on) => { el.style.visibility = on ? "visible" : "hidden"; };
  const rr = (o) => ({ x: o.x.x, y: o.y.x, w: Math.max(0, o.w.x), h: Math.max(0, o.h.x) });
  const setRect = (el, r) => {
    el.setAttribute("x", r.x.toFixed(1)); el.setAttribute("y", r.y.toFixed(1));
    el.setAttribute("width", r.w.toFixed(1)); el.setAttribute("height", r.h.toFixed(1));
  };

  function render(now) {
    // veil with a hole over the selected tile
    const va = clamp(veilA.x, 0, 1);
    if (va > 0.002) {
      const o = rr(hole), q = 4;
      veil.setAttribute("d", `M-2 -2H${FW + 2}V${FH + 2}H-2Z` +
        `M${o.x + q} ${o.y}H${o.x + o.w - q}Q${o.x + o.w} ${o.y} ${o.x + o.w} ${o.y + q}V${o.y + o.h - q}` +
        `Q${o.x + o.w} ${o.y + o.h} ${o.x + o.w - q} ${o.y + o.h}H${o.x + q}Q${o.x} ${o.y + o.h} ${o.x} ${o.y + o.h - q}` +
        `V${o.y + q}Q${o.x} ${o.y} ${o.x + q} ${o.y}Z`);
      setRect(selRing, { x: o.x + 1, y: o.y + 1, w: o.w - 2, h: o.h - 2 });
    }
    veil.style.opacity = va * 0.74;
    selRing.style.opacity = va;
    const ha = clamp(hov.a.x, 0, 1);
    if (ha > 0.002) { const o = rr(hov.r); setRect(hovRing, { x: o.x + 1, y: o.y + 1, w: o.w - 2, h: o.h - 2 }); }
    hovRing.style.opacity = ha;

    // peek
    const a = peek.a.x;
    const on = a > 0.004;
    vis(gPeek, on); vis(front, on); backs.forEach((d) => vis(d, on));
    if (on) {
      const sc = Math.max(0.05, 0.3 + 0.7 * a);
      const op = clamp(a * 1.6, 0, 1);
      const px = peek.p.x.x, py = peek.p.y.x;
      const rot = clamp(peek.p.x.v * 0.016, -9, 9);
      peekRect.setAttribute("transform", `translate(${px.toFixed(1)} ${py.toFixed(1)}) rotate(${rot.toFixed(2)}) scale(${sc.toFixed(3)})`);
      peekTail.setAttribute("d", tail(px, py, PEEK.w * sc, PEEK.h * sc, rot, PEEK.r * sc,
        { x: peek.tip.x.x, y: peek.tip.y.x }, { x: peek.p.x.v, y: peek.p.y.v }, 26 * sc));
      gPeek.style.opacity = op;
      front.style.opacity = op;
      front.style.transform = `translate(${px}px, ${py}px) rotate(${rot}deg) scale(${sc})`;
      peek.b.forEach((b, i) => {
        const r = (i ? 6 : -7) + clamp(b.x.v * 0.022, -12, 12);
        backs[i].style.opacity = op;
        backs[i].style.transform = `translate(${b.x.x}px, ${b.y.x}px) rotate(${r}deg) scale(${sc * 0.98})`;
      });
      // before -> after flip so the edit shows itself
      const showAfter = reduce.matches || ((now - peek.t0) % 2200) > 750;
      if (showAfter !== peek.after) {
        peek.after = showAfter;
        front.classList.toggle("after", showAfter);
        chip.textContent = showAfter ? "After" : "Before";
      }
    }

    // card
    const ca = cs.a.x;
    const cardOn = ca > 0.004 || cs.open;
    if (sheet) {                       // phones: a bottom sheet, styled by CSS
      vis(gCard, false);
      card.style.visibility = card.style.transform = card.style.opacity = "";
      return;
    }
    vis(gCard, cardOn);
    vis(card, cardOn);
    if (!cardOn) return;
    const o = rr(cs.r);
    setRect(cardRect, o);
    gCard.style.opacity = clamp(ca * 3, 0, 1);
    cardTail.setAttribute("d", tail(o.x + o.w / 2, o.y + o.h / 2, o.w, o.h, 0, 14,
      { x: cs.tip.x.x, y: cs.tip.y.x }, { x: cs.r.x.v, y: cs.r.y.v }, 34));
    const prog = clamp((o.w - cs.w0) / (CARD_W - cs.w0 || 1), 0, 1.2);
    card.style.opacity = clamp((prog - 0.55) / 0.4, 0, 1) * clamp(ca * 2, 0, 1);
    card.style.transform = `translate(${o.x * k}px, ${o.y * k}px) scale(${(k * o.w) / CARD_W}, ${(k * o.h) / (cs.H || 1)})`;
  }

  // ---------- layout ----------
  function layout() {
    const w = stage.clientWidth;
    k = w / FW;
    fx.style.transform = `scale(${k})`;
    const wasSheet = sheet;
    sheet = w < SHEET_BELOW;
    card.classList.toggle("sheet", sheet);
    if (sheet !== wasSheet && cs.open && !sheet) { cs.H = cIn.offsetHeight; snap(cs.r, cardTarget()); cs.w0 = 40; }
    if (cs.open && !sheet) measure();
    kick();
  }
  new ResizeObserver(layout).observe(stage);
  layout();

  // A one-off shimmer over the tiles the first time the figure is on screen.
  let seen = false;
  new IntersectionObserver((entries, obs) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    obs.disconnect();
    seen = true;
    if (!reduce.matches) under.classList.add("pulse");
    if (data) TILES.forEach((t) => preload(t, false));
  }, { threshold: 0.6 }).observe(stage);

  fetch("assets/examples/examples.json")
    .then((r) => r.json())
    .then((d) => {
      data = d;
      judgesEl.innerHTML = "";
      d.judges.forEach((j, i) => {
        const b = h("button", "", { type: "button", title: `EditJudge-Bench AUROC ${j.auroc.toFixed(1)}` });
        b.textContent = j.name;
        b.addEventListener("click", () => { cs.judge = i; renderScores(); });
        judgesEl.append(b);
      });
      stage.classList.add("ready");
      if (seen) TILES.forEach((t) => preload(t, false));
    })
    .catch(() => { fx.remove(); card.remove(); });
})();
