/* Simulate and optimise pages. All numbers come from the API; this file only collects
   inputs and displays results. API URLs come from data-* attributes rendered with
   url_for(), so they work behind a path-prefix proxy. */
(function () {
    "use strict";

    const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
    const num = (v, nd = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–"
        : Number(v).toLocaleString("en-NG", { minimumFractionDigits: nd, maximumFractionDigits: nd });
    const naira = (v) => "₦" + num(v, 0);
    const pct = (share, nd = 0) => num(share * 100, nd) + "%";
    const tip = (t) => t ? `<span class="tip" tabindex="0" title="${esc(t)}" aria-label="${esc(t)}">?</span>` : "";
    const list = (items, cls = "") => items && items.filter(Boolean).length
        ? `<ul class="explain-list">${items.filter(Boolean).map((t) => `<li class="${cls}">${esc(t)}</li>`).join("")}</ul>` : "";
    const val = (input) => (input && input.value !== "" ? Number(input.value) : null);
    const badge = (cls, text, title = "") => `<span class="badge ${esc(cls)}" title="${esc(title)}">${esc(text)}</span>`;
    const estBadge = (p) => p && p.kind === "predicted" ? " " + badge("kind-predicted", "estimated", p.note) : "";
    const probBar = (p) => `<div class="prob"><div class="bar"><span style="width:${Math.round((p || 0) * 100)}%"></span></div><b>${Math.round((p || 0) * 100)}%</b></div>`;
    const IMPORTANCE = { high: "High", medium: "Medium", low: "Low", off: "Off" };
    const LEVELS = ["high", "medium", "low", "off"];
    const POINTS = { high: 3, medium: 2, low: 1, off: 0 };
    const DISCLAIMER = "These are estimates from ingredient data to help you decide what to test. Taste, texture and "
        + "safety still need lab and product trials.";

    async function postJSON(url, body) {
        const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                       credentials: "same-origin", body: JSON.stringify(body) });
        let data = {};
        try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
        if (!res.ok) throw new Error(data.error || `Something went wrong (${res.status}). Please try again.`);
        return data;
    }

    // ------------------------------------------------------------ importance settings
    /* How much each property counts, per product: the defaults from the server plus the user's
       edits on this page. Only the changes are sent to the API. */
    function importanceState(cfg, product, overrides) {
        const defaults = {};
        cfg.products.forEach((p) => {
            defaults[p.id] = {};
            p.criteria.scored.forEach((c) => { defaults[p.id][c.property] = c.importance; });
            (p.criteria.optional || []).forEach((c) => { defaults[p.id][c.property] = "off"; });
        });
        const levels = JSON.parse(JSON.stringify(defaults));
        if (product && levels[product] && overrides) {
            Object.entries(overrides).forEach(([k, v]) => { if (k in levels[product] && v in POINTS) levels[product][k] = v; });
        }
        return {
            get: (pid) => levels[pid],
            defaultOf: (pid, key) => defaults[pid][key],
            set: (pid, key, v) => { levels[pid][key] = v; },
            reset: (pid) => { levels[pid] = { ...defaults[pid] }; },
            changes: (pid) => {
                const out = Object.fromEntries(Object.entries(levels[pid]).filter(([k, v]) => v !== defaults[pid][k]));
                return Object.keys(out).length ? out : null;
            },
        };
    }

    const encodePairs = (obj) => Object.entries(obj).map(([k, v]) => `${k}:${v}`).join(",");
    function parsePairs(text, asNumber) {
        const out = {};
        (text || "").split(",").forEach((part) => {
            const [k, v] = part.split(":");
            if (!k || !v) return;
            if (!asNumber) out[k.trim()] = v.trim();
            else if (Number(v) > 0) out[k.trim()] = Number(v);
        });
        return Object.keys(out).length ? out : null;
    }

    // ------------------------------------------------------------ form building
    function criteriaTable(product, imp) {
        const c = product.criteria || { scored: [], optional: [], not_scored: [] };
        const rows = [...c.scored, ...(c.optional || [])];
        const lv = imp.get(product.id);
        const total = rows.reduce((a, r) => a + POINTS[lv[r.property]], 0);
        const body = rows.map((r) => {
            const level = lv[r.property];
            const changed = level !== imp.defaultOf(product.id, r.property);
            return `<tr class="${level === "off" ? "row-off" : ""}">
                <td>${esc(r.label)}${r.direction === "lower" ? '<span class="range">lower is better</span>' : ""}</td>
                <td><select data-importance="${esc(r.property)}" aria-label="How much ${esc(r.label)} matters">
                    ${LEVELS.map((l) => `<option value="${l}" ${l === level ? "selected" : ""}>${IMPORTANCE[l]}</option>`).join("")}</select>
                    ${changed ? '<span class="range">changed</span>' : ""}</td>
                <td class="num">${level === "off" || !total ? "–" : pct(POINTS[level] / total)}</td>
                <td class="muted">${esc(r.why)}</td></tr>`;
        }).join("");
        const missing = c.not_scored.length ? `<p class="muted" style="font-size:13px">Not measured yet, so not counted:
            ${esc(c.not_scored.map((m) => m.label.toLowerCase()).join(", "))}.</p>` : "";
        return `<div class="criteria-box"><h3>What matters for ${esc(product.name.toLowerCase())}</h3>
            <p class="muted" style="font-size:13px">You can change how much each property counts. Off leaves it out.</p>
            <div class="table-wrap"><table class="data-table"><thead><tr><th>Property</th><th>How much it matters</th><th class="num">Weight</th><th>Why</th></tr></thead>
            <tbody>${body}</tbody></table></div>
            ${total ? "" : '<div class="error-box">Every property is off. Set at least one to High, Medium or Low.</div>'}
            ${imp.changes(product.id) ? '<div class="btn-row"><button type="button" class="btn small secondary" data-reset-importance>Back to the usual settings</button></div>' : ""}
            ${missing}</div>`;
    }

    function buildCommon(root, cfg, initial) {
        const pid0 = cfg.products.some((p) => p.id === initial.product) ? initial.product : cfg.products[0].id;
        root.querySelector("[data-products]").innerHTML = cfg.products.map((p) => `
            <label class="product-choice">
                <input type="radio" name="product" value="${esc(p.id)}" ${p.id === pid0 ? "checked" : ""}>
                <span><strong>${esc(p.name)}</strong><small>${esc(p.examples)}</small><small class="muted">${esc(p.summary)}</small></span>
            </label>`).join("");
        const sel = root.querySelector("[data-states]");
        sel.innerHTML = `<option value="">All of Nigeria</option>` + cfg.states.map((s) => `<option value="${esc(s.id)}">${esc(s.label)}${
            s.ingredients.length && s.prices ? " (local data and prices)" : s.ingredients.length ? " (local data)" : ""}</option>`).join("");
        if (initial.location) sel.value = initial.location;
        const imp = importanceState(cfg, pid0, initial.importance);
        const box = root.querySelector("[data-criteria]");
        const pidNow = () => (root.querySelector("input[name=product]:checked") || {}).value;
        const renderCriteria = () => { box.innerHTML = criteriaTable(cfg.products.find((p) => p.id === pidNow()), imp); };
        box.addEventListener("change", (e) => {
            if (!e.target.matches("[data-importance]")) return;
            imp.set(pidNow(), e.target.dataset.importance, e.target.value);
            renderCriteria();
        });
        box.addEventListener("click", (e) => {
            if (!e.target.matches("[data-reset-importance]")) return;
            imp.reset(pidNow());
            renderCriteria();
        });
        root.querySelectorAll("input[name=product]").forEach((r) => r.addEventListener("change", renderCriteria));
        renderCriteria();
        return imp;
    }

    const productOf = (root) => (root.querySelector("input[name=product]:checked") || {}).value;

    // ------------------------------------------------------------ result pieces
    function renderCertainty(c) {
        if (!c) return "";
        return `<div class="certainty-card ${esc(c.level)}">
            <div class="level">How sure are we? ${esc(c.label)}</div>
            <h3>${esc(c.headline)}</h3>
            <ul class="certainty-points">${c.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
            <details><summary>How we check this</summary><p class="muted">${esc(c.how || "")}</p></details>
            <details><summary>Good to know</summary><ul class="certainty-points">${(c.assumptions || []).map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>
        </div>`;
    }

    function propertyBars(contributions) {
        return `<div class="contrib">` + contributions.map((c) => `
            <span>${esc(c.label)}${c.best_of_selected ? ' <strong title="best of your proteins">★</strong>' : ""}</span>
            <span class="bar" title="Compared with the five proteins"><span style="width:${Math.round(100 * (c.relative || 0))}%"></span></span>
            <span class="muted">${num(c.value, c.unit === "g/g" ? 2 : 1)} ${esc(c.unit)}${c.quality === "checked" || c.quality === "rough" ? " (estimated)" : ""}</span>`).join("") + `</div>`;
    }

    function renderRanking(ranking, opts = {}) {
        if (!ranking || !ranking.length) return "";
        const chance = opts.chance && ranking.length > 1;
        return `<h3>${esc(opts.title)}</h3>
            <div class="table-wrap"><table class="data-table ranking-table"><thead><tr><th>#</th><th>Protein</th>
            <th class="num">Score ${tip("Out of 100: how well it suits this product compared with the other proteins.")}</th>
            ${chance ? `<th>Chance it is the best</th>` : ""}
            ${opts.price ? '<th class="num">Price</th>' : ""}<th>Strong and weak points</th></tr></thead><tbody>` +
            ranking.map((r) => `<tr class="${r.rank === 1 && ranking.length > 1 ? "rank-top" : ""}">
                <td><span class="rank-no">${r.rank}</span></td>
                <td><strong>${esc(r.name)}</strong></td>
                <td class="num"><strong>${num(r.score, 0)}</strong><span class="range">likely ${num(r.low, 0)}–${num(r.high, 0)}</span></td>
                ${chance ? `<td>${probBar(r.p_first)}</td>` : ""}
                ${opts.price ? `<td class="num">${r.price ? naira(r.price.value) + "/kg" + estBadge(r.price) : '<span class="muted">no price</span>'}</td>` : ""}
                <td class="muted">${esc(r.reason)}<details><summary>Each property</summary>${propertyBars(r.contributions)}</details></td>
            </tr>`).join("") + `</tbody></table></div>`;
    }

    function renderBlend(b, title, opts = {}) {
        const s = b.score, prot = b.protein, cost = b.cost;
        let html = `<h3>${esc(title)}</h3><div class="kpis">
            <div class="kpi"><small>Score</small><strong>${num(s.value, 0)} / 100</strong><span class="range">most likely ${num(s.low, 0)}–${num(s.high, 0)}</span></div>
            <div class="kpi"><small>Protein</small><strong>${prot ? num(prot.value, 0) : "–"} g</strong><span class="range">per 100 g of blend (dry)</span></div>
            <div class="kpi"><small>Cost</small><strong>${cost.available ? naira(cost.value) + "/kg" : "–"}</strong><span class="range">${cost.available
                ? (cost.batch !== null && cost.batch !== undefined ? `${naira(cost.batch)} for the batch` : "") +
                  (cost.estimated_prices && cost.estimated_prices.length ? " · uses estimated prices" : "")
                : esc(cost.reason)}</span></div>
        </div>`;
        html += `<div class="table-wrap"><table class="data-table"><thead><tr><th>Protein</th><th class="num">Share</th>
            <th class="num">Amount</th><th class="num">Price</th>${opts.limits ? "<th></th>" : ""}</tr></thead><tbody>` +
            b.components.map((c) => `<tr><td><strong>${esc(c.name)}</strong></td><td class="num"><strong>${pct(c.share, 1)}</strong></td>
                <td class="num">${c.kg !== null && c.kg !== undefined ? num(c.kg, 1) + " kg" : "–"}</td>
                <td class="num">${c.price ? naira(c.price.value) + "/kg" + estBadge(c.price) : "–"}</td>
                ${opts.limits ? `<td class="muted">${c.at_limit ? `at the ${esc(c.limit_source || "supplier maximum")} (${pct(c.max_share)})` : ""}</td>` : ""}</tr>`).join("") +
            "</tbody></table></div>";
        return html;
    }

    function renderComparison(r) {
        if (!r.comparison) return "";
        const b = r.comparison.baseline;
        const blendText = (x) => x.components.map((c) => `${pct(c.share)} ${esc(c.name)}`).join(", ");
        return `<h3>Compared with your blend</h3>` + list([r.comparison.text]) +
            `<div class="table-wrap"><table class="data-table"><thead><tr><th></th><th>Blend</th><th class="num">Score</th><th class="num">Protein (g)</th><th class="num">Cost</th></tr></thead><tbody>
            <tr><td>Your blend</td><td>${blendText(b)}</td><td class="num">${num(b.score.value, 0)}</td>
                <td class="num">${b.protein ? num(b.protein.value, 0) : "–"}</td><td class="num">${b.cost.available ? naira(b.cost.value) + "/kg" : "–"}</td></tr>
            <tr class="rank-top"><td>Recommended</td><td>${blendText(r.blend)}</td><td class="num"><strong>${num(r.blend.score.value, 0)}</strong></td>
                <td class="num">${r.blend.protein ? num(r.blend.protein.value, 0) : "–"}</td><td class="num">${r.blend.cost.available ? naira(r.blend.cost.value) + "/kg" : "–"}</td></tr>
            </tbody></table></div>`;
    }

    function renderLabPlan(plan, names) {
        if (!plan) return "";
        const ids = [...new Set([...plan.shortlist.map((s) => s.id), ...plan.runs.flatMap((r) => Object.keys(r.grams))])];
        const measure = plan.measure.map((m) => m.label.toLowerCase() + (m.no_data_yet ? " (not measured yet)" : ""));
        return `<h3>Next step: test in the lab</h3>
            <p><strong>Shortlist:</strong> ${plan.shortlist.map((s) => `${s.rank}. ${esc(s.name)}`).join(" · ")}</p>
            <p class="muted">${esc(plan.text)}</p>
            <div class="table-wrap"><table class="data-table"><thead><tr><th>Batch</th><th>Blend</th>
            ${ids.map((i) => `<th class="num">${esc(names[i] || i)} (g)</th>`).join("")}</tr></thead><tbody>` +
            plan.runs.map((r) => `<tr><td>${r.run}</td><td>${esc(r.label)}</td>${ids.map((i) => {
                const g = r.grams[i] || 0;
                return `<td class="num">${g ? num(g, Number.isInteger(g) ? 0 : 1) : "–"}</td>`;
            }).join("")}</tr>`).join("") +
            `</tbody></table></div>
            <p class="muted" style="font-size:13px">Test each batch for: ${esc(measure.join(", "))}.</p>`;
    }

    const where = (r) => `${esc(r.product.name)} · ${r.location ? esc(r.location) : "all of Nigeria"}`;
    const footer = () => `<p class="muted" style="font-size:12px;margin-top:18px">${esc(DISCLAIMER)}</p>`;

    function renderSimulation(r, e) {
        let html = `<p class="muted">${where(r)}</p><div class="headline">${esc(e.headline)}</div>`;
        html += renderCertainty(r.certainty);
        html += renderBlend(r.blend, "Your blend");
        if (e.binding) html += "<h3>Water and oil</h3>" + list([e.binding]);
        if (r.ranking.length > 1) html += renderRanking(r.ranking, { title: "What each protein adds" });
        return html + footer();
    }

    function renderOptimisation(data, names) {
        const r = data.result, e = data.explanation;
        let html = `<p class="muted">${where(r)}</p>`;
        if (!r.feasible) {
            html += `<div class="headline infeasible">${esc(e.headline)}</div>` + list(e.reasons, "warn");
            if (e.suggestions && e.suggestions.length) html += list(e.suggestions);
            html += renderCertainty(r.certainty);
            return html + renderRanking(r.ranking, { title: `Your proteins, ranked for ${r.product.name.toLowerCase()}`, chance: true, price: true }) + footer();
        }
        html += `<div class="headline">${esc(e.headline)}</div>`;
        html += renderCertainty(r.certainty);
        if (r.ranking.length > 1) {
            html += renderRanking(r.ranking, { title: `Your proteins, ranked for ${r.product.name.toLowerCase()}`, chance: true, price: true });
        }
        html += renderBlend(r.blend, "Recommended blend", { limits: true });
        html += renderComparison(r);
        if (e.blend && e.blend.length) html += "<h3>Why this blend</h3>" + list(e.blend);
        if (r.notes && r.notes.length) html += list(r.notes, "binding");
        if (e.binding) html += "<h3>Water and oil</h3>" + list([e.binding]);
        html += renderLabPlan(r.lab_plan, names);
        return html + footer();
    }

    // ------------------------------------------------------------ simulate page
    function initSimulate(root, cfg) {
        const imp = buildCommon(root, cfg, {});
        const form = root.querySelector("#simForm");
        const tbody = root.querySelector("[data-recipe]");
        const start = { soybean: 50, cowpea: 30, bambara_groundnut: 20 };
        tbody.innerHTML = cfg.ingredients.map((c) => `<tr>
            <td><strong>${esc(c.name)}</strong><span class="range">${esc(c.form)}</span></td>
            <td class="amount-cell"><input type="range" min="0" max="100" step="0.1" value="${start[c.id] || 0}" data-amount="${esc(c.id)}" aria-label="${esc(c.name)}">
                <input type="number" min="0" max="100" step="0.1" value="${start[c.id] || 0}" data-amount-number="${esc(c.id)}" aria-label="${esc(c.name)} %"></td></tr>`).join("");
        const totalEl = root.querySelector("#recipeTotal");
        const numbers = () => [...tbody.querySelectorAll("[data-amount-number]")];
        const amount = (el) => Math.max(0, Number(el.value) || 0);
        const recipe = () => Object.fromEntries(numbers().map((el) => [el.dataset.amountNumber, amount(el)]).filter(([, v]) => v > 0));
        const total = () => Math.round(numbers().reduce((a, el) => a + amount(el), 0) * 10) / 10;
        const updateTotal = () => {
            const t = total();
            const left = Math.round((100 - t) * 10) / 10;
            totalEl.textContent = left > 0 ? `Total: ${num(t, 1)}% (${num(left, 1)}% left to share)` : `Total: ${num(t, 1)}%`;
            totalEl.className = "share-total " + (Math.abs(t - 100) < 0.05 ? "good" : "bad");
        };
        /* The blend cannot add up to more than 100%: an amount is capped at what the other proteins leave. */
        tbody.addEventListener("input", (e) => {
            const id = e.target.dataset.amount || e.target.dataset.amountNumber;
            if (!id) return;
            const others = numbers().filter((el) => el.dataset.amountNumber !== id).reduce((a, el) => a + amount(el), 0);
            const cap = Math.max(0, Math.round((100 - others) * 10) / 10);
            const typed = e.target.value === "" ? 0 : Number(e.target.value);
            const v = Math.min(Math.max(Number.isFinite(typed) ? typed : 0, 0), cap);
            tbody.querySelectorAll(`[data-amount="${id}"], [data-amount-number="${id}"]`).forEach((el) => {
                if (el === e.target && (e.target.value === "" || v === typed)) return;  // leave what the user is typing
                el.value = v;
            });
            updateTotal();
        });
        root.querySelector("#evenSplit").addEventListener("click", () => {
            const left = Math.round((100 - total()) * 10) / 10;
            if (left <= 0) return;
            let targets = numbers().filter((el) => amount(el) > 0);
            if (!targets.length) targets = numbers();
            const each = Math.floor((left / targets.length) * 10) / 10;
            targets.forEach((el, k) => {
                const add = k === targets.length - 1 ? Math.round((left - each * (targets.length - 1)) * 10) / 10 : each;
                const v = Math.round((amount(el) + add) * 10) / 10;
                tbody.querySelectorAll(`[data-amount="${el.dataset.amountNumber}"], [data-amount-number="${el.dataset.amountNumber}"]`)
                    .forEach((x) => { x.value = v; });
            });
            updateTotal();
        });
        updateTotal();

        const out = root.querySelector("#result");
        const saveBtn = root.querySelector("#saveRun");
        const cmpBtn = root.querySelector("#addCompare");
        let last = null, lastBody = null;
        const scenarios = [];
        const body = () => ({
            product: productOf(root), location: root.querySelector("#location").value || null,
            recipe: recipe(), batch_kg: val(root.querySelector("#batch")), importance: imp.changes(productOf(root)),
        });
        const optimiseLink = (b) => {
            const q = new URLSearchParams({ product: b.product || "", location: b.location || "", baseline: encodePairs(b.recipe) });
            if (b.batch_kg) q.set("batch", b.batch_kg);
            if (b.importance) q.set("importance", encodePairs(b.importance));
            Object.keys(b.recipe).forEach((k) => q.append("ids", k));
            return `${root.dataset.optimisePage}?${q.toString()}`;
        };
        form.addEventListener("submit", async (ev) => {
            ev.preventDefault();
            out.innerHTML = `<div class="empty-state">Working it out…</div>`;
            saveBtn.disabled = cmpBtn.disabled = true;
            try {
                lastBody = body();
                const data = await postJSON(root.dataset.simulateUrl, lastBody);
                last = data.result;
                out.innerHTML = `<div class="btn-row" style="justify-content:flex-end"><a class="btn" href="${esc(optimiseLink(lastBody))}">Let the AI improve this blend &rarr;</a></div>`
                    + renderSimulation(data.result, data.explanation);
                saveBtn.disabled = cmpBtn.disabled = false;
                out.scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (err) {
                out.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
            }
        });
        const renderScenarios = () => {
            root.querySelector("#scenarios").hidden = !scenarios.length;
            root.querySelector("#scenarioTable").innerHTML = `<table class="data-table"><thead><tr><th>#</th><th>Product</th><th>State</th><th>Blend</th>
                <th class="num">Score</th><th class="num">Protein (g)</th><th class="num">Cost</th><th>How sure</th><th></th></tr></thead><tbody>` +
                scenarios.map((r, i) => `<tr><td>${i + 1}</td><td>${esc(r.product.name)}${r.importance_changes && r.importance_changes.length ? '<span class="range">your importance settings</span>' : ""}</td>
                    <td>${esc(r.location || "All of Nigeria")}</td>
                    <td>${r.blend.components.map((x) => `${pct(x.share)} ${esc(x.name)}`).join(", ")}</td>
                    <td class="num">${num(r.blend.score.value, 0)}</td>
                    <td class="num">${r.blend.protein ? num(r.blend.protein.value, 0) : "–"}</td>
                    <td class="num">${r.blend.cost.available ? naira(r.blend.cost.value) + "/kg" : "–"}</td>
                    <td>${badge(r.certainty.level, r.certainty.label)}</td>
                    <td><button class="btn small secondary" type="button" data-drop="${i}">Remove</button></td></tr>`).join("") + "</tbody></table>";
        };
        cmpBtn.addEventListener("click", () => { if (last) { scenarios.push(last); renderScenarios(); } });
        root.querySelector("#scenarios").addEventListener("click", (e) => {
            if (e.target.matches("[data-drop]")) { scenarios.splice(Number(e.target.dataset.drop), 1); renderScenarios(); }
        });
        bindSave(root, saveBtn, () => lastBody, root.dataset.simulateUrl);
    }

    // ------------------------------------------------------------ optimise page
    function initOptimise(root, cfg) {
        const params = new URLSearchParams(window.location.search);
        const imp = buildCommon(root, cfg, { product: params.get("product"), location: params.get("location"),
                                            importance: parsePairs(params.get("importance"), false) });
        const names = Object.fromEntries(cfg.ingredients.map((i) => [i.id, i.name]));
        const preselect = params.getAll("ids").filter((i) => names[i]);
        root.querySelector("[data-ingredient-choices]").innerHTML = cfg.ingredients.map((i) => `
            <label class="product-choice">
                <input type="checkbox" data-use="${esc(i.id)}" ${!preselect.length || preselect.includes(i.id) ? "checked" : ""}>
                <span><strong>${esc(i.name)}</strong><small class="muted">${esc(i.form)}</small></span>
            </label>`).join("");
        if (params.get("batch")) root.querySelector("#batch").value = params.get("batch");
        let baseline = parsePairs(params.get("baseline"), true);
        const note = root.querySelector("#baselineNote");
        const showBaseline = () => {
            note.innerHTML = baseline ? `<p class="muted" style="margin-top:10px">Comparing with your blend:
                ${esc(Object.entries(baseline).map(([k, v]) => `${num(v, 0)}% ${names[k] || k}`).join(", "))}.
                <button type="button" class="btn small secondary" id="clearBaseline">Don't compare</button></p>` : "";
            const clear = note.querySelector("#clearBaseline");
            if (clear) clear.addEventListener("click", () => { baseline = null; showBaseline(); });
        };
        showBaseline();
        const form = root.querySelector("#optForm");
        const out = root.querySelector("#result");
        const saveBtn = root.querySelector("#saveRun");
        let lastBody = null;
        const body = () => {
            const ids = [...root.querySelectorAll("[data-use]")].filter((c) => c.checked).map((c) => c.dataset.use);
            const usable = baseline && Object.keys(baseline).every((k) => ids.includes(k)) ? baseline : null;
            return {
                product: productOf(root), location: root.querySelector("#location").value || null,
                ingredient_ids: ids, batch_kg: val(root.querySelector("#batch")), budget: val(root.querySelector("#budget")),
                baseline: usable, importance: imp.changes(productOf(root)),
            };
        };
        const run = async () => {
            out.innerHTML = `<div class="empty-state">Ranking your proteins and finding the best blend…</div>`;
            saveBtn.disabled = true;
            try {
                lastBody = body();
                const data = await postJSON(root.dataset.optimiseUrl, lastBody);
                out.innerHTML = renderOptimisation(data, names);
                saveBtn.disabled = false;
                out.scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (err) {
                out.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
            }
        };
        form.addEventListener("submit", (ev) => { ev.preventDefault(); run(); });
        bindSave(root, saveBtn, () => lastBody, root.dataset.optimiseUrl);
        if (baseline) run();
    }

    function bindSave(root, btn, getBody, url) {
        btn.addEventListener("click", async () => {
            const b = getBody();
            if (!b) return;
            const name = prompt("Name for this blend:", "My blend");
            if (name === null) return;
            try {
                const saved = await postJSON(url, { ...b, save: true, name });
                root.querySelector("#savedNote").innerHTML =
                    `<a href="${esc(root.dataset.detailUrl.replace(/\/0$/, "/" + saved.saved_id))}">Saved. View →</a>`;
            } catch (err) { alert(err.message); }
        });
    }

    // ------------------------------------------------------------ saved detail page
    function initDetail(root) {
        const data = JSON.parse(root.querySelector("#savedData").textContent);
        const out = root.querySelector("#detailResult");
        if (!data.result || !data.result.engine) {
            out.innerHTML = `<p class="muted">This run was saved by an earlier version of NaijaProtein AI and cannot be
                shown any more. Run it again to see it with the current data.</p>`;
            return;
        }
        const names = {};
        (data.result.ranking || []).forEach((r) => { names[r.id] = r.name; });
        out.innerHTML = root.dataset.kind === "optimisation"
            ? renderOptimisation(data, names) : renderSimulation(data.result, data.explanation);
    }

    document.addEventListener("DOMContentLoaded", () => {
        const page = document.querySelector("[data-page]");
        const cfgEl = document.querySelector("#pageConfig");
        if (page && cfgEl) {
            const cfg = JSON.parse(cfgEl.textContent);
            if (page.dataset.page === "simulate") initSimulate(page, cfg);
            if (page.dataset.page === "optimise") initOptimise(page, cfg);
        }
        const det = document.querySelector("[data-saved-detail]");
        if (det) initDetail(det);
    });

    globalThis.NaijaProteinRender = { renderSimulation, renderOptimisation };
})();
