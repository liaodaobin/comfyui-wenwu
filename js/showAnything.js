import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const SIMPLE_TEXT_DEFAULTS = {
    text: "WenWu Simple Text",
    font_size: 32,
    font_color: "#f97316",
    font_weight: "normal",
    font_style: "normal",
    align: "left",
    bg_color: "transparent",
    padding: 12,
    border_radius: 8,
};

const COLOR_PRESETS = [
    "#000000", "#ffffff", "#6b7280", "#1f2937", "#ef4444",
    "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6",
    "#8b5cf6", "#ec4899",
];

function ensureSimpleTextDefaults(node) {
    node.properties ??= {};
    for (const [key, value] of Object.entries(SIMPLE_TEXT_DEFAULTS)) {
        if (node.properties[key] === undefined) node.properties[key] = value;
    }
}

function drawRoundRect(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, Math.min(w, h) / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

function getTextLines(text) {
    const value = String(text ?? "").replace(/\\n/g, "\n");
    return value.length ? value.split("\n") : [""];
}

function measureSimpleText(node) {
    ensureSimpleTextDefaults(node);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const fontSize = Math.max(8, Number(node.properties.font_size) || 32);
    const padding = Math.max(0, Number(node.properties.padding) || 0);
    ctx.font = `${node.properties.font_style} ${node.properties.font_weight} ${fontSize}px sans-serif`;
    const lines = getTextLines(node.properties.text);
    const width = Math.max(120, ...lines.map((line) => ctx.measureText(line).width)) + padding * 2;
    const height = Math.max(48, lines.length * fontSize * 1.25) + padding * 2;
    return [Math.ceil(width), Math.ceil(height)];
}

function openSimpleTextEditor(node) {
    ensureSimpleTextDefaults(node);

    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed; inset: 0; z-index: 10000; display: flex;
        align-items: center; justify-content: center; background: rgba(0,0,0,.45);
        font-family: system-ui, sans-serif;
    `;

    const panel = document.createElement("div");
    panel.style.cssText = `
        width: min(760px, calc(100vw - 48px)); max-height: calc(100vh - 48px);
        display: flex; flex-direction: column; background: #111827; color: #e5e7eb;
        border: 1px solid #374151; border-radius: 10px; overflow: hidden;
        box-shadow: 0 24px 80px rgba(0,0,0,.45);
    `;

    const header = document.createElement("div");
    header.style.cssText = "padding: 16px 20px; border-bottom: 1px solid #374151;";
    header.innerHTML = `<div style="font-weight:700;font-size:18px">编辑文本 (SimpleText)</div><div style="color:#9ca3af;margin-top:4px">支持多行，可调整字号、颜色、背景、对齐和边距</div>`;

    const body = document.createElement("div");
    body.style.cssText = "padding: 18px 20px; overflow:auto; display:grid; gap:14px;";

    const text = document.createElement("textarea");
    text.value = String(node.properties.text ?? "");
    text.style.cssText = `
        width: 100%; min-height: 150px; resize: vertical; box-sizing: border-box;
        background: #0b1220; color: #f9fafb; border: 1px solid #374151;
        border-radius: 8px; padding: 12px; font: 14px ui-monospace, monospace;
    `;
    body.appendChild(text);

    const fields = document.createElement("div");
    fields.style.cssText = "display:grid; grid-template-columns: 130px 1fr; gap:12px 14px; align-items:center;";
    body.appendChild(fields);

    const values = {};
    function row(label, input) {
        const l = document.createElement("label");
        l.textContent = label;
        l.style.color = "#cbd5e1";
        fields.append(l, input);
        return input;
    }
    function input(type, key, attrs = {}) {
        const el = document.createElement("input");
        el.type = type;
        el.value = node.properties[key];
        Object.assign(el, attrs);
        el.style.cssText = "height:32px;background:#0b1220;color:#f9fafb;border:1px solid #374151;border-radius:6px;padding:0 8px;";
        values[key] = el;
        return el;
    }
    function select(key, options) {
        const el = document.createElement("select");
        el.style.cssText = "height:34px;background:#0b1220;color:#f9fafb;border:1px solid #374151;border-radius:6px;padding:0 8px;";
        for (const [value, label] of options) {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = label;
            el.appendChild(opt);
        }
        el.value = node.properties[key];
        values[key] = el;
        return el;
    }

    row("字号", input("number", "font_size", { min: 8, max: 96 }));
    row("文字颜色", input("color", "font_color"));
    row("字重", select("font_weight", [["normal", "正常"], ["bold", "粗体"]]));
    row("字形", select("font_style", [["normal", "正常"], ["italic", "倾斜"]]));
    row("排列方式", select("align", [["left", "左"], ["center", "中"], ["right", "右"]]));
    row("背景颜色", input("color", "bg_color"));
    const transparent = input("checkbox", "bg_transparent");
    transparent.checked = node.properties.bg_color === "transparent";
    transparent.value = "true";
    row("背景透明", transparent);
    row("内边距", input("number", "padding", { min: 0, max: 64 }));
    row("圆角", input("number", "border_radius", { min: 0, max: 48 }));

    const swatches = document.createElement("div");
    swatches.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;grid-column:2;";
    for (const color of COLOR_PRESETS) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.title = color;
        btn.style.cssText = `width:24px;height:24px;border-radius:5px;border:1px solid #475569;background:${color};cursor:pointer;`;
        btn.onclick = () => { values.font_color.value = color; };
        swatches.appendChild(btn);
    }
    fields.appendChild(document.createElement("div"));
    fields.appendChild(swatches);

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:flex-end;gap:10px;padding:14px 20px;border-top:1px solid #374151;";
    const cancel = document.createElement("button");
    cancel.textContent = "取消";
    const ok = document.createElement("button");
    ok.textContent = "确定";
    for (const btn of [cancel, ok]) {
        btn.style.cssText = "min-width:86px;height:36px;border:0;border-radius:8px;padding:0 14px;cursor:pointer;";
    }
    cancel.style.background = "#374151";
    cancel.style.color = "#f9fafb";
    ok.style.background = "#2563eb";
    ok.style.color = "#ffffff";
    footer.append(cancel, ok);

    cancel.onclick = () => overlay.remove();
    ok.onclick = () => {
        node.properties.text = text.value;
        node.properties.font_size = Math.max(8, Math.min(96, Number(values.font_size.value) || 32));
        node.properties.font_color = values.font_color.value;
        node.properties.font_weight = values.font_weight.value;
        node.properties.font_style = values.font_style.value;
        node.properties.align = values.align.value;
        node.properties.bg_color = transparent.checked ? "transparent" : values.bg_color.value;
        node.properties.padding = Math.max(0, Math.min(64, Number(values.padding.value) || 0));
        node.properties.border_radius = Math.max(0, Math.min(48, Number(values.border_radius.value) || 0));
        node.size = measureSimpleText(node);
        node.setDirtyCanvas?.(true, true);
        overlay.remove();
    };

    overlay.addEventListener("keydown", (event) => {
        if (event.key === "Escape") overlay.remove();
        event.stopPropagation();
    });

    panel.append(header, body, footer);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    text.focus();
}

function installWenWuSimpleText(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        ensureSimpleTextDefaults(this);
        this.size = measureSimpleText(this);
    };

    nodeType.prototype.onDblClick = function () {
        openSimpleTextEditor(this);
        return true;
    };

    const getMenuOptions = nodeType.prototype.getMenuOptions;
    nodeType.prototype.getMenuOptions = function (canvas) {
        const base = getMenuOptions ? getMenuOptions.call(this, canvas) : [];
        return [
            { content: "编辑文本 (Edit Text)", callback: () => openSimpleTextEditor(this) },
            null,
            ...base,
        ];
    };

    nodeType.prototype.computeSize = function () {
        return measureSimpleText(this);
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
        ensureSimpleTextDefaults(this);
        const props = this.properties;
        const [w, h] = this.size ?? measureSimpleText(this);
        const padding = Math.max(0, Number(props.padding) || 0);
        const fontSize = Math.max(8, Number(props.font_size) || 32);
        const lines = getTextLines(props.text);

        ctx.save();
        if (props.bg_color !== "transparent") {
            ctx.fillStyle = props.bg_color;
            drawRoundRect(ctx, 0, 0, w, h, Number(props.border_radius) || 0);
            ctx.fill();
        }
        ctx.fillStyle = props.font_color;
        ctx.font = `${props.font_style} ${props.font_weight} ${fontSize}px sans-serif`;
        ctx.textBaseline = "top";
        ctx.textAlign = props.align;
        const x = props.align === "center" ? w / 2 : props.align === "right" ? w - padding : padding;
        const lineHeight = fontSize * 1.25;
        lines.forEach((line, index) => ctx.fillText(line, x, padding + index * lineHeight));
        ctx.restore();
    };
}

app.registerExtension({
    name: "ShowAnything|WenWu",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.category !== "WenWu/Prompt") {
            return;
        }

        if (nodeData.name === "WenWuSimpleTextNode") {
            installWenWuSimpleText(nodeType);
            return;
        }

        if (nodeData.name === "WenWuShowAndSaveAnything") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (!this.textWidget) {
                    this.textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    this.textWidget.inputEl.readOnly = true;
                    this.textWidget.inputEl.style.border = "none";
                    this.textWidget.inputEl.style.backgroundColor = "transparent";
                }

                const raw = message?.text ?? message?.ui?.text ?? "";
                const text = Array.isArray(raw) ? (raw[0] ?? "") : raw;
                this.textWidget.value = String(text);
                this.textWidget.inputEl.value = this.textWidget.value;
            };
        }
    },
});
