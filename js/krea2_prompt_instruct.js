import { app } from "../../../scripts/app.js";

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function sanitizeLegacyInputs(node) {
    for (const name of ["images", "parameters"]) {
        const index = (node.inputs || []).findIndex((input) => input.name === name);
        if (index >= 0) {
            node.removeInput(index);
        }
    }
}

function normalizeNumberWidget(node, name, fallback, min = null, max = null) {
    const widget = findWidget(node, name);
    if (!widget) return;
    let value = Number(widget.value);
    if (!Number.isFinite(value)) value = fallback;
    if (min != null) value = Math.max(min, value);
    if (max != null) value = Math.min(max, value);
    if (widget.value !== value) {
        widget.value = value;
        widget.callback?.(value);
    }
}

function sanitizeLegacyWidgetValues(node) {
    normalizeNumberWidget(node, "max_frames", 24, 2, 1024);
    normalizeNumberWidget(node, "max_size", 256, 128, 16384);
    normalizeNumberWidget(node, "seed", 0, 0);
}

function setWidgetVisible(node, widget, visible) {
    if (!widget) return;
    if (!widget._wenwuOriginalComputeSize) {
        widget._wenwuOriginalComputeSize = widget.computeSize;
    }

    widget.hidden = !visible;
    widget.options = widget.options || {};
    widget.options.hidden = !visible;
    widget.computeSize = visible
        ? widget._wenwuOriginalComputeSize
        : () => [0, 0];

    if (widget.element) {
        widget.element.style.display = visible ? "" : "none";
    }

    const width = node.size?.[0] || 360;
    node.setSize(node.computeSize ? node.computeSize() : [width, node.size?.[1] || 200]);
    app.graph.setDirtyCanvas(true, true);
}

function updateStyleImageVisibility(node) {
    const toggle = findWidget(node, "style_reference");
    const image = findWidget(node, "style_image");
    setWidgetVisible(node, image, Boolean(toggle?.value));
}

app.registerExtension({
    name: "WenWu.Krea2PromptInstruct",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WenWuKrea2PromptInstruct") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            sanitizeLegacyInputs(this);
            sanitizeLegacyWidgetValues(this);

            const toggle = findWidget(this, "style_reference");
            if (toggle && !toggle._wenwuKrea2Wrapped) {
                const originalCallback = toggle.callback;
                toggle.callback = (...args) => {
                    originalCallback?.apply(toggle, args);
                    updateStyleImageVisibility(this);
                };
                toggle._wenwuKrea2Wrapped = true;
            }

            updateStyleImageVisibility(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            sanitizeLegacyInputs(this);
            sanitizeLegacyWidgetValues(this);
            updateStyleImageVisibility(this);
        };
    },
});
