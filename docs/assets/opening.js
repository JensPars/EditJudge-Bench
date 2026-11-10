/* Opening block of the project page: "What changed?"

   The visitor picks the instruction that matches a before/after pair, then sees
   how the judges scored every instruction. Three rounds, then a small scoreboard.

   Data: the "opening" list in assets/examples/examples.json, images in
   assets/examples/opening/ (scripts/build_page_examples.py). */
(() => {
  "use strict";
  const root = document.getElementById("opening");
  if (!root) return;

  const BASE = "assets/examples/";
  const IMG = `${BASE}opening/`;
  const JUDGES = [0, 1]; // indices into examples.json "judges": GPT-4o, GPT-5.6-sol
  // Forced-choice accuracy: Table 1 (MC column) and the human study.
  const STATS = [
    { name: "People", v: 95.2, cls: "people", note: "Human study: 5 participants × 100 pairs" },
    { name: "GPT-5.6-sol", v: 71.5, cls: "", note: "Best judge · VIEScore template · all 1,500 pairs" },
    { name: "GPT-4o", v: 35.8, cls: "", note: "VIEScore template · all 1,500 pairs" },
    { name: "Chance", v: 21, cls: "chance", approx: true, note: "Picking one of the instructions at random" },
  ];

  const OK = `<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M3.6 8.4l2.9 2.9 5.9-6.2" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const NO = `<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>`;
  const ARROW = `<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M3 8h9.5M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const DOWN = `<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M8 3v9.5M4 8.5l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (v) => (v == null ? "–" : Number.isInteger(v) ? String(v) : v.toFixed(1));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const shuffle = (a) => {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };

  root.innerHTML = `
    <div class="op-card">
      <div class="op-frame">
        <div class="op-pair">
          <img class="op-b" alt="The scene before the edit" draggable="false">
          <img class="op-a" alt="The scene after the edit" draggable="false">
        </div>
        <div class="op-seg" role="group" aria-label="Image shown">
          <button type="button" data-s="b" aria-pressed="true">Before</button>
          <button type="button" data-s="a" aria-pressed="false">After</button>
        </div>
      </div>
      <div class="op-panel" tabindex="-1"></div>
    </div>
    <div class="op-strip">
      <span class="op-cap">Ranks the true instruction first</span>
      <div class="op-stats">${STATS.map((s) => `
        <div class="op-stat ${s.cls}" title="${esc(s.note)}">
          <span class="l">${esc(s.name)}</span><span class="v">${s.approx ? "≈" : ""}${s.v}%</span>
          <span class="t"><i style="--v:${s.v}%"></i></span>
        </div>`).join("")}
      </div>
    </div>
    <p class="op-live" role="status" aria-live="polite"></p>`;

  const card = root.querySelector(".op-card");
  const panel = root.querySelector(".op-panel");
  const strip = root.querySelector(".op-strip");
  const live = root.querySelector(".op-live");

  let data = null;
  let rounds = []; // the three examples, in order
  let visible = true;

  /* Before/after frame: one large image that alternates, so the edit is easy to spot. */
  const frame = (() => {
    const el = root.querySelector(".op-frame");
    const imgB = el.querySelector(".op-b");
    const imgA = el.querySelector(".op-a");
    const segs = [...el.querySelectorAll(".op-seg button")];
    let side = "b";
    let over = false;
    let holdUntil = 0;
    let timer = 0;
    let token = 0;
    const paint = () => {
      el.classList.toggle("after", side === "a");
      segs.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === side)));
    };
    const tick = () => {
      clearTimeout(timer);
      if (reduce || over || !visible) return;
      const base = side === "b" ? 1300 : 1700;
      timer = setTimeout(() => {
        side = side === "b" ? "a" : "b";
        paint();
        tick();
      }, Math.max(base, holdUntil - Date.now()));
    };
    const set = (s) => {
      side = s;
      holdUntil = Date.now() + 4000; // a manual choice stays put for a moment
      paint();
      tick();
    };
    segs.forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); set(b.dataset.s); }));
    el.addEventListener("click", () => set(side === "b" ? "a" : "b"));
    el.addEventListener("pointerenter", (e) => { if (e.pointerType === "mouse") { over = true; tick(); } });
    el.addEventListener("pointerleave", () => { over = false; tick(); });
    const show = (id) => {
      const my = ++token;
      const b = new Image();
      const a = new Image();
      b.src = `${IMG}${id}_b.webp`;
      a.src = `${IMG}${id}_a.webp`;
      el.classList.add("swap");
      clearTimeout(timer);
      // Wait for both images (load event; decode() can stall in a background tab), 2 s at most.
      const ready = (im) => new Promise((r) => { if (im.complete) r(); else im.onload = im.onerror = r; });
      const loaded = Promise.race([Promise.all([ready(b), ready(a)]), new Promise((r) => setTimeout(r, 2000))]);
      Promise.all([loaded, new Promise((r) => setTimeout(r, 180))]).then(() => {
        if (my !== token) return;
        imgB.src = b.src;
        imgA.src = a.src;
        side = "b";
        holdUntil = 0;
        paint();
        el.classList.remove("swap");
        tick();
      });
    };
    return { show, tick };
  })();

  /* Which instruction a judge "picks": the one it scores highest. */
  const verdict = (ex, j) => {
    const s = ex.prompts.map((p) => p.s[j]);
    const max = Math.max(...s.filter((v) => v != null));
    const tops = s.map((v) => v === max);
    const n = tops.filter(Boolean).length;
    return { s, tops, right: tops[0] && n === 1, tie: tops[0] && n > 1 };
  };
  const icon = (ok) => `<span class="op-res ${ok ? "ok" : "no"}" role="img" aria-label="${ok ? "right" : "wrong"}">${ok ? OK : NO}</span>`;

  /* The quiz. */
  const quiz = {
    start() {
      this.round = 0;
      this.picks = [];
      this.judge = 0;
      strip.classList.remove("show");
      this.ask();
    },
    dots() {
      return `<div class="op-dots" aria-hidden="true">${rounds.map((_, n) => {
        const done = n < this.picks.length;
        return `<i class="${done ? (this.picks[n] ? "ok" : "no") : n === this.round ? "now" : ""}"></i>`;
      }).join("")}</div>`;
    },
    ask() {
      const ex = (this.ex = rounds[this.round]);
      this.picked = null;
      frame.show(ex.id);
      const order = shuffle(ex.prompts.map((_, i) => i));
      const last = this.round === rounds.length - 1;
      panel.innerHTML = `
        <div class="op-head"><p class="op-q">What changed?</p>${this.dots()}</div>
        <div class="op-opts">${order.map((i, n) => `
          <button type="button" class="op-opt" data-i="${i}" style="--n:${n}">
            <i class="op-fill"></i><span class="op-mark"></span>
            <span class="op-txt">${esc(ex.prompts[i].text)}</span><span class="op-num"></span>
          </button>`).join("")}
        </div>
        <div class="op-foot">
          <div class="op-chips">${this.chips(true)}</div>
          <button type="button" class="op-next">${last ? "Result" : "Next"}${ARROW}</button>
        </div>`;
      panel.querySelectorAll(".op-opt").forEach((b) => b.addEventListener("click", () => this.pick(+b.dataset.i)));
      panel.querySelector(".op-next").addEventListener("click", () => {
        if (last) this.done();
        else { this.round++; this.ask(); }
        panel.focus({ preventScroll: true });
      });
    },
    // Rendered (hidden) together with the question, so the reveal does not change the height.
    chips(ok) {
      return `<span class="op-chip ${ok ? "ok" : "no"}">You${ok ? OK : NO}</span>` + JUDGES.map((j) => {
        const r = verdict(this.ex, j).right;
        return `<button type="button" class="op-chip ${r ? "ok" : "no"}" data-j="${j}" title="Show this judge's scores">${esc(data.judges[j].name)}${r ? OK : NO}</button>`;
      }).join("");
    },
    pick(i) {
      if (this.picked != null) return;
      this.picked = i;
      const ok = i === 0; // prompts[0] is the true instruction
      this.picks[this.round] = ok;
      panel.querySelector(".op-opts").classList.add("shown");
      panel.querySelectorAll(".op-opt").forEach((b) => {
        const k = +b.dataset.i;
        b.disabled = true;
        if (k === 0) { b.classList.add("true"); b.querySelector(".op-mark").innerHTML = OK; }
        else if (k === i) { b.classList.add("miss"); b.querySelector(".op-mark").innerHTML = NO; }
      });
      const chips = panel.querySelector(".op-chips");
      chips.innerHTML = this.chips(ok);
      chips.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { this.judge = +b.dataset.j; this.paint(); }));
      panel.querySelector(".op-dots").outerHTML = this.dots();
      panel.querySelector(".op-foot").classList.add("show");
      strip.classList.add("show");
      this.paint();
      panel.querySelector(".op-next").focus({ preventScroll: true });
      live.textContent = `${ok ? "Correct." : "Not quite."} ${panel.querySelector(".op-q").textContent}`;
    },
    paint() {
      const v = verdict(this.ex, this.judge);
      const name = esc(data.judges[this.judge].name);
      panel.querySelectorAll(".op-opt").forEach((b) => {
        const s = v.s[+b.dataset.i];
        b.classList.toggle("top", v.tops[+b.dataset.i]);
        b.querySelector(".op-fill").style.width = `${s == null ? 0 : (100 * s) / data.scale}%`;
        b.querySelector(".op-num").textContent = fmt(s);
      });
      const q = panel.querySelector(".op-q");
      q.className = `op-q ${v.right ? "good" : "bad"}`;
      q.innerHTML = v.right ? `${name} gets it <b>right</b>.` : v.tie ? `${name} <b>can’t decide</b>.` : `${name} is <b>fooled</b>.`;
      panel.querySelectorAll(".op-chips button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.j === this.judge)));
    },
    done() {
      const n = this.picks.filter(Boolean).length;
      const rows = [["You", this.picks]].concat(JUDGES.map((j) => [data.judges[j].name, rounds.map((ex) => verdict(ex, j).right)]));
      panel.innerHTML = `
        <div class="op-head"><p class="op-q">You got ${n} of ${rounds.length}.</p></div>
        <table class="op-board">
          <thead><tr><td></td>${rounds.map((ex) => `<th><img src="${IMG}${ex.id}_a.webp" alt=""></th>`).join("")}<td></td></tr></thead>
          <tbody>${rows.map(([name, r]) => `
            <tr><th scope="row">${esc(name)}</th>${r.map((ok) => `<td>${icon(ok)}</td>`).join("")}<td class="op-tally">${r.filter(Boolean).length}/${r.length}</td></tr>`).join("")}
          </tbody>
        </table>
        <div class="op-foot show">
          <button type="button" class="op-again">Again</button>
          <a class="op-next" href="#benchmark">Browse the edits${DOWN}</a>
        </div>`;
      panel.querySelector(".op-again").addEventListener("click", () => { this.start(); panel.focus({ preventScroll: true }); });
      live.textContent = `You got ${n} of ${rounds.length}.`;
    },
  };

  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => {
      visible = es[0].isIntersecting;
      if (visible) frame.tick();
    }).observe(card);
  }

  fetch(`${BASE}examples.json`)
    .then((r) => r.json())
    .then((d) => {
      data = d;
      rounds = d.opening;
      rounds.forEach((ex) => ["b", "a"].forEach((side) => { new Image().src = `${IMG}${ex.id}_${side}.webp`; }));
      quiz.start();
    })
    .catch(() => { root.closest("section").hidden = true; });
})();
