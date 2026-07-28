import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const HIDDEN_WIDGETS = new Set([
    "video_frames", "reference_video_frames", "image_detail", "temperature",
    "top_p", "max_tokens", "timeout", "model_name", "clear_context",
    "uploaded_images_json",
    "resolution",
]);

const FIXED_VALUES = {
    video_frames: 3,
    reference_video_frames: 0,
    image_detail: "auto",
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 4096,
    timeout: 30,
    model_name: "ComfyUI-llama-cpp",
    clear_context: true,
};

const MAX_IMAGES = 6;

if (!window.__wenwuReferencePasteInterceptorInstalled) {
    window.__wenwuReferencePasteInterceptorInstalled = true;
    window.addEventListener("paste", (event) => {
        window.__wenwuActiveReferencePasteHandler?.(event);
    }, true);
    window.addEventListener("pointerdown", (event) => {
        const activePanel = window.__wenwuActiveReferencePastePanel;
        if (activePanel && !activePanel.contains(event.target)) {
            window.__wenwuActiveReferencePasteHandler = null;
            window.__wenwuActiveReferencePastePanel = null;
        }
    }, true);
}

function hideWidget(widget) {
    if (!widget || widget._wenwuHidden) return;
    widget._wenwuHidden = true;
    widget.hidden = true;
    widget.options = widget.options || {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    for (const element of [widget.inputEl, widget.element].filter(Boolean)) {
        element.style.display = "none";
        element.hidden = true;
    }
}

function safeImages(widget) {
    try {
        const value = JSON.parse(widget?.value || "[]");
        return Array.isArray(value) ? value.filter((item) => typeof item === "string").slice(0, MAX_IMAGES) : [];
    } catch {
        return [];
    }
}

function viewUrl(filename) {
    const parts = String(filename).replaceAll("\\\\", "/").split("/");
    const name = parts.pop();
    const subfolder = parts.join("/");
    const params = new URLSearchParams({ filename: name, type: "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

async function uploadImage(file) {
    const mimeExtension = file.type?.split("/")[1]?.replace("jpeg", "jpg");
    const nameExtension = file.name?.includes(".") ? file.name.split(".").pop() : "";
    const extension = (mimeExtension || nameExtension || "png").replace(/[^a-zA-Z0-9]/g, "");
    const uniqueName = `wenwu-${Date.now()}-${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}.${extension}`;
    const form = new FormData();
    form.append("image", file, uniqueName);
    form.append("type", "input");
    form.append("overwrite", "false");
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) throw new Error(`Upload failed (${response.status})`);
    const result = await response.json();
    return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}

function createImagePanel(node, stateWidget) {
    const panel = document.createElement("div");
    panel.tabIndex = 0;
    panel.style.cssText = [
        "box-sizing:border-box", "width:100%", "max-width:100%", "min-width:0", "height:250px", "padding:7px", "border:1px solid #4b5563",
        "border-radius:7px", "background:#303030", "color:#d4d4d4", "font:12px sans-serif",
        "outline:none", "overflow:hidden",
    ].join(";");

    const hint = document.createElement("div");
    hint.textContent = "参考图（最多 6 张） · 点击/拖入图片 · 选中此框后 Ctrl+V 粘贴";
    hint.style.cssText = "height:24px;line-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#b8b8b8";

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(2,100px);gap:6px;height:206px";
    panel.append(hint, grid);

    let images = safeImages(stateWidget);
    let draggedIndex = null;
    let selectedIndex = null;

    const save = () => {
        stateWidget.value = JSON.stringify(images);
        stateWidget.callback?.(stateWidget.value);
        node.setDirtyCanvas?.(true, true);
    };

    const addFiles = async (files, insertAt = images.length) => {
        const imageFiles = [...files].filter((file) => file.type?.startsWith("image/"));
        if (!imageFiles.length) return;
        panel.style.opacity = "0.65";
        try {
            for (const file of imageFiles) {
                if (images.length >= MAX_IMAGES) break;
                const uploaded = await uploadImage(file);
                images.splice(Math.min(insertAt, images.length), 0, uploaded);
                insertAt += 1;
            }
            save();
            render();
        } catch (error) {
            console.error("WenWu image upload failed", error);
            alert(`文武节点图片上传失败：${error.message}`);
        } finally {
            panel.style.opacity = "1";
        }
    };

    const chooseFiles = (insertAt) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = true;
        input.onchange = () => addFiles(input.files || [], insertAt);
        input.click();
    };

    const placeImageAt = async (imageFile, index) => {
        panel.style.opacity = "0.65";
        try {
            const uploaded = await uploadImage(imageFile);
            if (images[index]) {
                images[index] = uploaded;
            } else {
                images.splice(Math.min(index, images.length), 0, uploaded);
                images = images.slice(0, MAX_IMAGES);
            }
            save();
            render();
        } catch (error) {
            console.error("WenWu clipboard image paste failed", error);
            alert(`无法读取剪贴板图片：${error.message}\n也可以先点击目标格子，再按 Ctrl+V。`);
        } finally {
            panel.style.opacity = "1";
        }
    };

    const pasteImageAt = async (index) => {
        if (!navigator.clipboard?.read) {
            alert("当前浏览器不支持按钮读取剪贴板，请先选中目标格子，再按 Ctrl+V。");
            return;
        }
        try {
            const clipboardItems = await navigator.clipboard.read();
            for (const item of clipboardItems) {
                const imageType = item.types.find((type) => type.startsWith("image/"));
                if (!imageType) continue;
                const blob = await item.getType(imageType);
                const extension = imageType.split("/")[1]?.replace("jpeg", "jpg") || "png";
                const imageFile = new File(
                    [blob],
                    `wenwu-paste-${Date.now()}.${extension}`,
                    { type: imageType },
                );
                await placeImageAt(imageFile, index);
                return;
            }
            alert("剪贴板里没有图片。请先复制图片，再点击“粘贴”。");
        } catch (error) {
            console.error("WenWu clipboard image paste failed", error);
            alert(`无法读取剪贴板图片：${error.message}\n也可以先选中目标格子，再按 Ctrl+V。`);
        }
    };

    const updateSelection = () => {
        [...grid.children].forEach((slot, index) => {
            slot.style.border = selectedIndex === index
                ? "2px solid #60a5fa"
                : "1px dashed #4b5563";
        });
    };

    const handleSelectedSlotPaste = (event) => {
        const files = [...(event.clipboardData?.items || [])]
            .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
            .map((item) => item.getAsFile()).filter(Boolean);
        if (!files.length || selectedIndex === null) return;
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        placeImageAt(files[0], selectedIndex);
    };

    const activateSlotPaste = () => {
        window.__wenwuActiveReferencePasteHandler = handleSelectedSlotPaste;
        window.__wenwuActiveReferencePastePanel = panel;
        panel.focus({ preventScroll: true });
    };

    const render = () => {
        grid.replaceChildren();
        for (let index = 0; index < MAX_IMAGES; index += 1) {
            const slot = document.createElement("div");
            slot.dataset.index = String(index);
            slot.style.cssText = [
                "position:relative", "display:flex", "align-items:center", "justify-content:center",
                "min-width:0", "overflow:hidden",
                selectedIndex === index ? "border:2px solid #60a5fa" : "border:1px dashed #4b5563",
                "border-radius:5px",
                "background:#202020", "cursor:pointer", "user-select:none",
            ].join(";");
            slot.title = images[index] || `单击选中，双击打开文件夹`;
            slot.onclick = () => {
                selectedIndex = index;
                activateSlotPaste();
                updateSelection();
            };
            slot.ondblclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                selectedIndex = index;
                chooseFiles(index);
            };
            slot.ondragstart = () => { draggedIndex = index; };
            slot.ondragover = (event) => { event.preventDefault(); slot.style.borderColor = "#60a5fa"; };
            slot.ondragleave = () => { slot.style.borderColor = "#4b5563"; };
            slot.ondrop = (event) => {
                event.preventDefault();
                event.stopPropagation();
                slot.style.borderColor = "#4b5563";
                if (event.dataTransfer?.files?.length) {
                    addFiles(event.dataTransfer.files, index);
                } else if (draggedIndex !== null && images[draggedIndex]) {
                    const [moved] = images.splice(draggedIndex, 1);
                    images.splice(index, 0, moved);
                    images = images.slice(0, MAX_IMAGES);
                    draggedIndex = null;
                    save();
                    render();
                }
            };

            if (images[index]) {
                slot.draggable = true;
                const img = document.createElement("div");
                img.style.cssText = [
                    "position:absolute", "inset:0",
                    "background-color:#181818", "background-repeat:no-repeat",
                    "background-position:center center", "background-size:contain",
                    "pointer-events:none",
                ].join(";");
                img.style.backgroundImage = `url("${viewUrl(images[index])}")`;
                const remove = document.createElement("button");
                remove.type = "button";
                remove.textContent = "×";
                remove.title = "从文武节点移除这张图片";
                remove.style.cssText = "position:absolute;top:3px;right:3px;width:22px;height:22px;padding:0;border:0;border-radius:50%;background:#111c;color:#fff;font:18px/20px sans-serif;cursor:pointer";
                remove.onclick = (event) => {
                    event.stopPropagation();
                    images.splice(index, 1);
                    save();
                    render();
                };
                slot.append(img, remove);
            } else {
                const label = document.createElement("span");
                label.textContent = `图片 ${index + 1}\n＋`;
                label.style.cssText = "white-space:pre;text-align:center;color:#888;line-height:22px";
                slot.append(label);
            }
            const paste = document.createElement("button");
            paste.type = "button";
            paste.textContent = "粘贴";
            paste.title = images[index] ? "粘贴并替换这张图片" : "从剪贴板粘贴图片到这个位置";
            paste.style.cssText = [
                "position:absolute", "left:4px", "bottom:4px", "height:22px", "padding:0 7px",
                "border:1px solid #666", "border-radius:4px", "background:#111c", "color:#eee",
                "font:11px/20px sans-serif", "cursor:pointer", "z-index:2",
            ].join(";");
            paste.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                selectedIndex = index;
                activateSlotPaste();
                render();
                pasteImageAt(index);
            };
            slot.append(paste);
            grid.append(slot);
        }
    };

    panel.addEventListener("dragover", (event) => event.preventDefault());
    panel.addEventListener("drop", (event) => {
        if (event.target === panel || event.target === grid) {
            event.preventDefault();
            addFiles(event.dataTransfer?.files || []);
        }
    });
    panel.addEventListener("paste", (event) => {
        const files = [...(event.clipboardData?.items || [])]
            .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
            .map((item) => item.getAsFile()).filter(Boolean);
        if (files.length) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const targetIndex = selectedIndex ?? Math.min(images.length, MAX_IMAGES - 1);
            placeImageAt(files[0], targetIndex);
        }
    }, true);
    node._wenwuRefreshImages = () => {
        images = safeImages(stateWidget);
        render();
    };
    node._wenwuSyncGallerySize = () => {
        panel.style.width = `${Math.max(120, Number(node.size?.[0] || 430) - 20)}px`;
        panel.style.maxWidth = panel.style.width;
    };
    node._wenwuSyncGallerySize();
    render();
    return panel;
}

function applyCompactLayout(node) {
    const resolutionValue = String(node.widgets?.find((widget) => widget.name === "resolution")?.value || "832*480");
    const resolutionMatch = resolutionValue.match(/(\d+)\s*[*×x]\s*(\d+)/i);
    const fallbackWidth = Number(resolutionMatch?.[1]) || 832;
    const fallbackHeight = Number(resolutionMatch?.[2]) || 480;
    for (const [name, fallback] of [["custom_width", fallbackWidth], ["custom_height", fallbackHeight]]) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) continue;
        const normalize = (value) => {
            const numeric = Number(value);
            return Number.isFinite(numeric) && numeric >= 16 ? Math.round(numeric) : fallback;
        };
        if (widget.value === "" || widget.value == null || !Number.isFinite(Number(widget.value))) {
            widget.value = normalize(widget.value);
            widget.callback?.(widget.value);
        }
        widget.serializeValue = async function () {
            const value = normalize(this.value);
            if (this.value !== value) this.value = value;
            return value;
        };
    }

    for (const widget of node.widgets || []) {
        if (HIDDEN_WIDGETS.has(widget.name)) {
            const fixedValue = FIXED_VALUES[widget.name];
            if (fixedValue !== undefined && widget.value !== fixedValue) {
                widget.value = fixedValue;
                widget.callback?.(fixedValue);
            }
            hideWidget(widget);
        }
    }
    const size = node.computeSize?.();
    if (size) node.setSize([Math.max(node.size?.[0] || 430, 430), Math.max(size[1], 610)]);
    node._wenwuSyncGallerySize?.();
    node.setDirtyCanvas?.(true, true);
}

function installImagePanel(node) {
    if (node._wenwuImagePanelInstalled) return;
    const stateWidget = node.widgets?.find((widget) => widget.name === "uploaded_images_json");
    if (!stateWidget) return;
    node._wenwuImagePanelInstalled = true;
    const panel = createImagePanel(node, stateWidget);
    node.addDOMWidget("wenwu_reference_gallery", "div", panel, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 250,
    });
}

function installPromptPreview(node) {
    if (node._wenwuPromptPreviewInstalled) return;
    node._wenwuPromptPreviewInstalled = true;
    node.properties ??= {};

    const container = document.createElement("div");
    container.style.cssText = `
        box-sizing: border-box; width: 100%; max-width: 100%; min-width: 0;
        padding: 8px 10px 10px; overflow: hidden;
        display: flex; flex-direction: column; gap: 6px;
        background: rgba(12, 18, 28, .88); border: 1px solid #3b4658;
        border-radius: 7px; color: #dbeafe; font-family: system-ui, sans-serif;
    `;

    const title = document.createElement("div");
    title.textContent = "AI 推理提示词";
    title.style.cssText = "font-size:12px;font-weight:700;color:#93c5fd;";

    const textarea = document.createElement("textarea");
    textarea.readOnly = true;
    textarea.placeholder = "执行后在这里显示 AI 模型生成的最终提示词";
    textarea.value = String(node.properties.wenwu_last_prompt || "");
    textarea.style.cssText = `
        box-sizing: border-box; width: 100%; max-width: 100%; min-width: 0;
        height: 136px; resize: vertical;
        padding: 9px 10px; border: 1px solid #334155; border-radius: 6px;
        outline: none; background: #090f19; color: #f8fafc;
        font: 12px/1.5 ui-monospace, Consolas, monospace;
        white-space: pre-wrap; overflow-wrap: anywhere;
    `;

    container.append(title, textarea);
    node.addDOMWidget("wenwu_prompt_preview", "div", container, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 190,
    });
    node._wenwuPromptPreviewElement = textarea;
    node._wenwuPromptPreviewContainer = container;
    node._wenwuSyncPromptPreviewSize = () => {
        const width = Math.max(120, Number(node.size?.[0] || 430) - 20);
        container.style.width = `${width}px`;
        container.style.maxWidth = `${width}px`;
        textarea.style.width = "100%";
        textarea.style.maxWidth = "100%";
    };
    node._wenwuSyncPromptPreviewSize();
}

app.registerExtension({
    name: "WenWu.PromptGeneratorIntegratedImages",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WenWuPromptGenerator") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            installImagePanel(this);
            installPromptPreview(this);
            applyCompactLayout(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            installImagePanel(this);
            installPromptPreview(this);
            if (this._wenwuPromptPreviewElement) {
                this._wenwuPromptPreviewElement.value = String(this.properties?.wenwu_last_prompt || "");
            }
            this._wenwuRefreshImages?.();
            applyCompactLayout(this);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            installPromptPreview(this);
            const raw = message?.wenwu_prompt ?? message?.ui?.wenwu_prompt ?? "";
            const prompt = Array.isArray(raw) ? (raw[0] ?? "") : raw;
            this.properties ??= {};
            this.properties.wenwu_last_prompt = String(prompt);
            if (this._wenwuPromptPreviewElement) {
                this._wenwuPromptPreviewElement.value = String(prompt);
                this._wenwuPromptPreviewElement.scrollTop = 0;
            }
            const size = this.computeSize?.();
            if (size) this.setSize([Math.max(this.size?.[0] || 430, 430), Math.max(size[1], 800)]);
            this.setDirtyCanvas?.(true, true);
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            this._wenwuSyncGallerySize?.();
            this._wenwuSyncPromptPreviewSize?.();
        };
    },
});
