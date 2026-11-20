/* Simulation and optimisation pages. All numbers come from the API; this file
   only collects inputs and displays results. API URLs come from data-* attributes
   rendered with url_for(), so they work behind a path-prefix proxy. */
(function () {
    "use strict";

    const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
    const num = (v, nd = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–"
        : Number(v).toLocaleString("en-NG", { minimumFractionDigits: nd, maximumFractionDigits: nd });
    const naira = (v, nd = 0) => "₦" + num(v, nd);
    const pct = (share, nd = 1) => num(share * 100, nd) + "%";
    const KIND = { measured: "measured", calculated: "calculated", predicted: "estimated", unavailable: "not available",
                   definition: "by definition" };
    const kind = (k) => `<span class="badge kind-${esc(k)}">${esc(KIND[k] || k)}</span>`;
    const tip = (t) => t ? `<span class="tip" tabindex="0" title="${esc(t)}" aria-label="${esc(t)}">?</span>` : "";
    const list = (items, cls = "") => items && items.length
        ? `<ul class="explain-list">${items.map((t) => `<li class="${cls}">${esc(t)}</li>`).join("")}</ul>` : "";
    const val = (input) => (input && input.value !== "" ? Number(input.value) : null);

    async function postJSON(url, body) {
        const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                       credentials: "same-origin", body: JSON.stringify(body) });
        let data = {};
        try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
        if (!res.ok) throw new Error(data.error || `Something went wrong (${res.status}).`);
        return data;
    }

    // ------------------------------------------------------------ form building
    function buildCommon(root, cfg) {
        root.querySelector("[data-products]").innerHTML = cfg.products.map((p, i) => `
            <label class="product-choice">
                <input type="radio" name="product" value="${esc(p.id)}" ${i === 0 ? "checked" : ""}>
                <span><strong>${esc(p.name)}</strong><small>${esc(p.examples)}</small><small class="muted">${esc(p.summary)}</small></span>
            </label>`).join("");
        root.querySelector("[data-states]").innerHTML = `<option value="">All of Nigeria (all data combined)</option>` +
            cfg.states.map((s) => `<option value="${esc(s.id)}">${esc(s.label)}${s.ingredients.length ? " • data" : ""}${s.prices ? " • prices" : ""}</option>`).join("");
        root.querySelector("[data-nutrition-targets]").innerHTML = cfg.nutrition
            .filter((n) => ["protein", "fat", "moisture", "fibre", "carbohydrate", "ash"].includes(n.key))
            .map((n) => `<tr><td>${esc(n.label)} ${tip(n.plain)}<span class="range">g per 100 g of mix</span></td>
                <td><input type="number" min="0" max="100" step="any" data-target="${esc(n.key)}" data-bound="min" aria-label="${esc(n.label)} at least"></td>
                <td><input type="number" min="0" max="100" step="any" data-target="${esc(n.key)}" data-bound="max" aria-label="${esc(n.label)} at most"></td></tr>`).join("");
        const renderFunctional = () => {
            const pid = (root.querySelector("input[name=product]:checked") || {}).value;
            const product = cfg.products.find((p) => p.id === pid);
            const relevant = new Map(product.relevant.map((r) => [r.property, r]));
            const props = cfg.functional.filter((f) => relevant.has(f.key));
            const defaults = cfg.product_targets[pid] || [];
            const anyModel = props.some((f) => cfg.blend_models[f.key]);
            root.querySelector("[data-functional-targets]").innerHTML = `
                <p class="muted">For ${esc(product.name.toLowerCase())}, these matter: ${esc(product.needs.join(", "))}.</p>
                <div class="table-wrap"><table class="data-table targets-table"><thead><tr><th>Property</th><th>Why it matters</th><th>At least</th><th>At most</th></tr></thead><tbody>` +
                props.map((f) => {
                    const r = relevant.get(f.key);
                    const ok = cfg.blend_models[f.key];
                    const d = defaults.find((t) => t.property === f.key);
                    const input = (b) => ok ? `<input type="number" min="0" step="any" data-target="${esc(f.key)}" data-bound="${b}"
                        value="${d && d[b] !== null ? d[b] : ""}" aria-label="${esc(f.label)} ${b}">` : "";
                    return `<tr><td>${esc(f.label)} ${tip(f.plain)}<span class="range">${esc(f.unit)}</span></td>
                        <td class="muted">${esc(r.why)}${d ? `<span class="range">Default from: ${esc(d.source)}</span>` : ""}</td>
                        ${ok ? `<td>${input("min")}</td><td>${input("max")}</td>`
                             : `<td colspan="2" class="muted">Can't be set yet: needs measured data on mixes.</td>`}</tr>`;
                }).join("") + `</tbody></table></div>` +
                (anyModel ? "" : `<p class="muted">How these properties behave in a mix can't be calculated from the ingredients
                    alone, and there is no measured data on mixes yet. The results will show each ingredient's own values for guidance.</p>`) +
                (defaults.length ? "" : `<p class="muted" style="font-size:12px">No published target values for this product are
                    recorded, so none are pre-filled.</p>`);
        };
        root.querySelectorAll("input[name=product]").forEach((r) => r.addEventListener("change", renderFunctional));
        renderFunctional();
    }

    function readTargets(root) {
        const targets = {};
        root.querySelectorAll("[data-target]").forEach((el) => {
            if (el.disabled || el.value === "") return;
            const t = (targets[el.dataset.target] = targets[el.dataset.target] || {});
            t[el.dataset.bound] = Number(el.value);
        });
        return targets;
    }

    // ------------------------------------------------------------ result rendering
    function renderComposition(r) {
        const keys = ["protein", "fat", "carbohydrate", "fibre", "ash", "moisture"];
        const rows = keys.map((k) => r.mix[k]).filter(Boolean).map((m) => `<tr><td>${esc(m.label)} ${tip(m.plain)}</td>
            <td class="num"><strong>${m.available ? num(m.value) : "–"}</strong></td>
            <td class="num">${r.dry_protein_blend[m.property] !== null && r.dry_protein_blend[m.property] !== undefined ? num(r.dry_protein_blend[m.property]) : "–"}</td>
            <td>${kind(m.kind)}${m.uses_predicted && m.uses_predicted.length ? `<span class="range">uses estimated values for ${esc(m.uses_predicted.join(", "))}</span>` : ""}
                ${!m.available ? `<span class="range">${esc(m.reason)}</span>` : ""}</td></tr>`).join("");
        const extra = Object.values(r.mix).filter((m) => !keys.includes(m.property) && m.available);
        return `<h3>Nutrition</h3><div class="table-wrap"><table class="data-table"><thead><tr><th>Nutrient</th>
            <th class="num">Per 100 g of mix</th><th class="num">Protein ingredients only, dry ${tip("The protein ingredients without added water or oil, on a dry-matter basis.")}</th><th>Type</th></tr></thead>
            <tbody>${rows}${extra.map((m) => `<tr><td>${esc(m.label)}</td><td class="num">${num(m.value)}</td><td class="num">${num(r.dry_protein_blend[m.property])}</td><td>${kind(m.kind)}</td></tr>`).join("")}</tbody></table></div>`;
    }

    function renderFunctional(r) {
        const rows = Object.values(r.functional).filter((f) => f.relevant || f.available || f.ingredient_values.length);
        if (!rows.length) return "";
        rows.sort((a, b) => (b.relevant - a.relevant));
        return `<h3>How it is likely to behave</h3><p class="muted">These properties change when ingredients are mixed, so they are not averaged.</p>
            <div class="table-wrap"><table class="data-table"><thead><tr><th>Property</th><th>For your mix</th><th>Each ingredient on its own</th></tr></thead><tbody>` +
            rows.map((f) => `<tr class="${f.relevant ? "relevant-row" : ""}"><td>${esc(f.label)} ${tip(f.plain)}<span class="range">${esc(f.unit)}</span>
                    ${f.relevant ? `<span class="relevance">${esc(f.why)}</span>` : ""}</td>
                <td>${f.available ? `<strong>${num(f.value)}</strong> ${kind("predicted")}<span class="range">likely ${num(f.low)}–${num(f.high)}</span>`
                                  : `${kind("unavailable")}<span class="range">Not enough measured data on mixes.</span>`}</td>
                <td class="muted">${f.ingredient_values.map((v) => `${esc(v.name)} ${num(v.value)}`).join("; ") || "no data"}${f.missing_for.length ? `<span class="range">no data: ${esc(f.missing_for.join(", "))}</span>` : ""}</td></tr>`).join("") +
            "</tbody></table></div>";
    }

    function renderTargets(targets) {
        if (!targets || !targets.length) return "";
        const icon = { met: "✓", not_met: "✗", not_checked: "–" };
        return `<h3>Your targets</h3><ul class="explain-list">` + targets.map((t) =>
            `<li class="target-${esc(t.status)}"><strong>${icon[t.status]}</strong> ${esc(t.text)}</li>`).join("") + "</ul>";
    }

    function renderContributions(rows) {
        if (!rows || !rows.length) return "";
        const cell = (r, k) => r[k] === undefined ? "–" : `${num(r[k])} <span class="range">${num(r[k + "_pct"], 0)}%</span>`;
        return `<h3>What each ingredient contributes</h3><div class="table-wrap"><table class="data-table"><thead><tr>
            <th>Ingredient</th><th class="num">Amount</th><th class="num">Protein (g/100 g)</th><th class="num">Fat</th>
            <th class="num">Fibre</th><th class="num">Cost (₦/kg)</th></tr></thead><tbody>` +
            rows.map((r) => `<tr><td>${esc(r.name)}</td><td class="num">${pct(r.share)}</td><td class="num">${cell(r, "protein")}</td>
                <td class="num">${cell(r, "fat")}</td><td class="num">${cell(r, "fibre")}</td><td class="num">${cell(r, "cost")}</td></tr>`).join("") +
            "</tbody></table></div>";
    }

    function renderEvidence(items) {
        if (!items || !items.length) return "";
        return `<details class="evidence"><summary>Technical details: every value used (${items.length})</summary>
            <div class="table-wrap"><table class="data-table"><thead><tr><th>Ingredient</th><th>Value</th><th class="num">Used</th>
            <th class="num">Range</th><th>Type</th><th>Where from</th><th>Source</th></tr></thead><tbody>` +
            items.map((e) => `<tr><td>${esc(e.ingredient)}</td><td>${esc(e.quantity)}</td>
                <td class="num">${num(e.value)} <span class="range">${esc(e.unit)}</span></td>
                <td class="num">${num(e.range[0])}–${num(e.range[1])}<span class="range">${e.n_records} record(s)</span></td>
                <td>${kind(e.kind)}</td><td class="muted">${esc(e.location_note)}<span class="range">${esc((e.states || []).join(", "))}</span></td>
                <td class="muted">${esc((e.sources || []).join("; "))}${e.prediction ? `<span class="range">${esc(e.prediction.method)}</span>` : ""}</td></tr>`).join("") +
            "</tbody></table></div></details>";
    }

    function renderSimulation(r, e) {
        const prot = r.mix.protein, fat = r.mix.fat, c = r.cost;
        let html = `<p class="muted">${esc(r.product.name)}${r.location ? " · " + esc(r.location) : " · all of Nigeria"}</p>
            <div class="headline">${esc(e.headline)}</div>
            <div class="kpis">
                <div class="kpi"><small>Protein</small><strong>${prot.available ? num(prot.value, 1) : "–"}</strong><span class="range">g per 100 g of mix</span></div>
                <div class="kpi"><small>Fat</small><strong>${fat.available ? num(fat.value, 1) : "–"}</strong><span class="range">g per 100 g of mix</span></div>
                <div class="kpi"><small>Cost</small><strong>${c.available ? naira(c.value) : "–"}</strong><span class="range">${c.available ? "per kg" + (c.batch_cost !== undefined ? ` · ${naira(c.batch_cost)} per batch` : "") : esc(c.reason)}</span></div>
            </div>`;
        html += renderTargets(r.targets);
        html += `<h3>Your recipe</h3><div class="table-wrap"><table class="data-table"><thead><tr><th>Component</th><th class="num">% of mix</th><th class="num">kg</th></tr></thead><tbody>` +
            r.components.map((x) => `<tr><td>${esc(x.name)}</td><td class="num">${pct(x.share)}</td><td class="num">${x.kg !== null ? num(x.kg, 1) : "–"}</td></tr>`).join("") + "</tbody></table></div>";
        html += renderComposition(r) + renderFunctional(r) + renderContributions(e.contributions || r.contributions);
        if (e.product && e.product.length) html += `<h3>Why these ingredients suit ${esc(r.product.name.toLowerCase())}</h3>` + list(e.product.map((p) => p.text));
        if (e.water_oil && e.water_oil.length) html += "<h3>Water and oil</h3>" + list(e.water_oil);
        if (e.availability && e.availability.length) html += "<h3>Stock</h3>" + list(e.availability);
        if (e.data && e.data.length) html += "<h3>Where the numbers come from</h3>" + list(e.data);
        if (e.limitations && e.limitations.length) html += "<h3>Limitations</h3>" + list(e.limitations, "warn");
        return html + renderEvidence(e.evidence);
    }

    function renderInfeasible(r, e) {
        let html = `<div class="headline infeasible">${esc(e.headline)}</div><p class="muted">${esc(e.summary || "")}</p>`;
        html += "<h3>Why</h3>" + list(e.reasons || [], "warn");
        if (e.suggestions && e.suggestions.length) html += "<h3>What you could change</h3>" + list(e.suggestions);
        if (e.location && e.location.text) html += "<h3>Effect of the state</h3>" + list([e.location.text]);
        return html;
    }

    function renderOptimisation(data) {
        const r = data.result, e = data.explanation;
        if (!r.feasible) return renderInfeasible(r, e);
        let html = `<div class="headline">${esc(e.headline)}</div>
            <div class="table-wrap"><table class="data-table"><thead><tr><th>Component</th><th class="num">% of mix</th>
            <th class="num">% of protein ingredients</th><th class="num">kg</th><th class="num">Cost</th><th>Limit</th></tr></thead><tbody>` +
            r.components.map((c) => `<tr><td>${esc(c.name)}</td><td class="num"><strong>${pct(c.share)}</strong></td>
                <td class="num">${c.share_of_protein_blend !== undefined ? pct(c.share_of_protein_blend) : "–"}</td>
                <td class="num">${num(c.kg, 1)}</td><td class="num">${c.price_ngn_per_kg !== null ? naira(c.cost_ngn) : "not costed"}</td>
                <td class="muted">${pct(c.min_share, 0)}–${pct(c.max_share, 0)}${c.limit_source ? `<span class="range">${esc(c.limit_source)}</span>` : ""}</td></tr>`).join("") +
            "</tbody></table></div>";
        html += "<h3>Why this recipe</h3>" + list(e.ingredients.map((i) => i.text));
        if (e.constraints.length) html += `<h3>How your requirements were met</h3><ul class="explain-list">` +
            e.constraints.map((c) => `<li class="${c.binding ? "binding" : ""}">${esc(c.text)}</li>`).join("") + "</ul>";
        if (e.location && e.location.text) html += "<h3>Effect of the state</h3>" + list([e.location.text]);
        if (data.profile) {
            html += `<h2 style="margin-top:26px">The recipe in detail</h2>`;
            html += renderSimulation(data.profile, { ...e, headline: "Calculated from the recipe above.",
                                                    water_oil: e.water_oil, contributions: data.profile.contributions });
        }
        return html + `<p class="muted" style="font-size:12px">Solved with ${esc(r.solver)}.</p>`;
    }

    // ------------------------------------------------------------ simulate page
    function initSimulate(root, cfg) {
        buildCommon(root, cfg);
        const form = root.querySelector("#simForm");
        const tbody = root.querySelector("[data-recipe]");
        const components = [...cfg.ingredients.map((i) => ({ id: i.id, name: i.name, note: i.form })),
                            { id: "water", name: "Added water", note: "How much water would you like to add?" },
                            { id: "oil", name: "Edible oil", note: "How much oil would you like to add?" }];
        const start = { soybean: 30, defatted_groundnut: 20, cowpea: 10, water: 35, oil: 5 };
        tbody.innerHTML = components.map((c) => `<tr class="${c.id === "water" || c.id === "oil" ? "support-row" : ""}">
            <td><strong>${esc(c.name)}</strong><span class="range">${esc(c.note)}</span></td>
            <td class="amount-cell"><input type="range" min="0" max="100" step="0.5" value="${start[c.id] || 0}" data-amount="${esc(c.id)}" aria-label="${esc(c.name)}">
                <input type="number" min="0" max="100" step="0.5" value="${start[c.id] || 0}" data-amount-number="${esc(c.id)}" aria-label="${esc(c.name)} %"></td></tr>`).join("");
        const totalEl = root.querySelector("#recipeTotal");
        const recipe = () => Object.fromEntries([...tbody.querySelectorAll("[data-amount-number]")]
            .map((el) => [el.dataset.amountNumber, Number(el.value) || 0]).filter(([, v]) => v > 0));
        const updateTotal = () => {
            const t = Object.values(recipe()).reduce((a, b) => a + b, 0);
            totalEl.textContent = `Total: ${num(t, 1)}%`;
            totalEl.className = "share-total " + (Math.abs(t - 100) < 0.05 ? "good" : "bad");
        };
        tbody.addEventListener("input", (e) => {
            const id = e.target.dataset.amount || e.target.dataset.amountNumber;
            if (!id) return;
            tbody.querySelectorAll(`[data-amount="${id}"], [data-amount-number="${id}"]`).forEach((el) => { if (el !== e.target) el.value = e.target.value; });
            updateTotal();
        });
        root.querySelector("#evenSplit").addEventListener("click", () => {
            const inputs = [...tbody.querySelectorAll("[data-amount-number]")];
            const water = inputs.filter((i) => ["water", "oil"].includes(i.dataset.amountNumber));
            const prot = inputs.filter((i) => !water.includes(i) && Number(i.value) > 0);
            const fixed = water.reduce((a, i) => a + (Number(i.value) || 0), 0);
            if (!prot.length || fixed >= 100) return;
            const each = Math.round(((100 - fixed) / prot.length) * 10) / 10;
            prot.forEach((i, k) => { i.value = k === prot.length - 1 ? (100 - fixed - each * (prot.length - 1)).toFixed(1) : each; i.dispatchEvent(new Event("input", { bubbles: true })); });
        });
        updateTotal();

        const out = root.querySelector("#result");
        const saveBtn = root.querySelector("#saveRun");
        const cmpBtn = root.querySelector("#addCompare");
        let last = null, lastBody = null;
        const scenarios = [];
        const body = () => ({
            product: (root.querySelector("input[name=product]:checked") || {}).value,
            location: root.querySelector("#location").value || null,
            recipe: recipe(), batch_kg: val(root.querySelector("#batch")), targets: readTargets(root),
            water_price: val(root.querySelector("#waterPrice")), oil_price: val(root.querySelector("#oilPrice")),
        });
        form.addEventListener("submit", async (ev) => {
            ev.preventDefault();
            out.innerHTML = `<div class="empty-state">Calculating…</div>`;
            saveBtn.disabled = cmpBtn.disabled = true;
            try {
                lastBody = body();
                const data = await postJSON(root.dataset.simulateUrl, lastBody);
                last = data.result;
                out.innerHTML = renderSimulation(data.result, data.explanation);
                saveBtn.disabled = cmpBtn.disabled = false;
                out.scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (err) {
                out.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
            }
        });
        const renderScenarios = () => {
            root.querySelector("#scenarios").hidden = !scenarios.length;
            const v = (m) => m && m.available ? num(m.value, 1) : "–";
            root.querySelector("#scenarioTable").innerHTML = `<table class="data-table"><thead><tr><th>#</th><th>Product</th><th>State</th><th>Recipe</th>
                <th class="num">Protein</th><th class="num">Fat</th><th class="num">Moisture</th><th class="num">₦/kg</th><th>Targets met</th><th></th></tr></thead><tbody>` +
                scenarios.map((r, i) => `<tr><td>${i + 1}</td><td>${esc(r.product.name)}</td><td>${esc(r.location || "Nigeria")}</td>
                    <td>${r.components.map((x) => `${pct(x.share, 0)} ${esc(x.name)}`).join(", ")}</td>
                    <td class="num">${v(r.mix.protein)}</td><td class="num">${v(r.mix.fat)}</td><td class="num">${v(r.mix.moisture)}</td>
                    <td class="num">${r.cost.available ? naira(r.cost.value) : "–"}</td>
                    <td>${r.targets.length ? `${r.targets.filter((t) => t.status === "met").length} of ${r.targets.length}` : "–"}</td>
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
        buildCommon(root, cfg);
        root.querySelector("[data-ingredient-limits]").innerHTML = cfg.ingredients.map((i) => `<tr>
            <td><input type="checkbox" data-use="${esc(i.id)}" checked aria-label="Use ${esc(i.name)}"></td>
            <td>${esc(i.name)}<span class="range">${esc(i.form)}</span></td>
            <td><input type="number" min="0" max="100" step="any" data-lim-min="${esc(i.id)}" placeholder="0"></td>
            <td><input type="number" min="0" max="100" step="any" data-lim-max="${esc(i.id)}" placeholder="supplier limit"></td></tr>`).join("");
        const protein = root.querySelector('[data-target="protein"][data-bound="min"]');
        if (protein) protein.value = 15;
        const form = root.querySelector("#optForm");
        const out = root.querySelector("#result");
        const saveBtn = root.querySelector("#saveRun");
        let lastBody = null;
        const range = (use, lo, hi) => root.querySelector(use).checked
            ? { min: val(root.querySelector(lo)), max: val(root.querySelector(hi)) } : null;
        const body = () => {
            const ids = [...root.querySelectorAll("[data-use]")].filter((c) => c.checked).map((c) => c.dataset.use);
            const limits = Object.fromEntries(ids.map((id) => [id, {
                min: val(root.querySelector(`[data-lim-min="${id}"]`)), max: val(root.querySelector(`[data-lim-max="${id}"]`)) }]));
            return {
                product: (root.querySelector("input[name=product]:checked") || {}).value,
                location: root.querySelector("#location").value || null,
                ingredient_ids: ids, limits, targets: readTargets(root),
                water: range("#useWater", "#waterMin", "#waterMax"), oil: range("#useOil", "#oilMin", "#oilMax"),
                limit_water_to_holding: root.querySelector("#holdingRule").checked,
                water_price: val(root.querySelector("#waterPrice")), oil_price: val(root.querySelector("#oilPrice")),
                batch_kg: val(root.querySelector("#batch")), budget: val(root.querySelector("#budget")),
            };
        };
        form.addEventListener("submit", async (ev) => {
            ev.preventDefault();
            out.innerHTML = `<div class="empty-state">Searching for the best recipe…</div>`;
            saveBtn.disabled = true;
            try {
                lastBody = body();
                const data = await postJSON(root.dataset.optimiseUrl, lastBody);
                out.innerHTML = renderOptimisation(data);
                saveBtn.disabled = false;
                out.scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (err) {
                out.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
            }
        });
        bindSave(root, saveBtn, () => lastBody, root.dataset.optimiseUrl);
    }

    function bindSave(root, btn, getBody, url) {
        btn.addEventListener("click", async () => {
            const b = getBody();
            if (!b) return;
            const name = prompt("Name for this recipe:", "My recipe");
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
        if (root.dataset.kind === "optimisation") {
            out.innerHTML = renderOptimisation({ result: data.result, profile: data.result.profile, explanation: data.explanation });
        } else {
            out.innerHTML = renderSimulation(data.result, data.explanation);
        }
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
