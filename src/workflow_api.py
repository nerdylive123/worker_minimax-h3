"""Convert ComfyUI GUI-format workflows to API ``/prompt`` format.

GUI format = what the frontend saves (nodes/links/definitions.subgraphs).
API format = what POST /prompt accepts: {node_id: {class_type, inputs}}.

The bundled MiniMax-H3 templates keep the sampler graph inside one subgraph
instance. User knobs (prompt, width, height, duration, model filenames, turbo
LoRA) are the subgraph's widget inputs. We set them on the instance's
``widgets_values``, then expand subgraphs, collapsing the boundary
input/output routing nodes so only real compute nodes remain.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

# Friendly-name -> index into the MiniMax-H3 subgraph instance widgets_values.
SUBGRAPH_WIDGET_ORDER = [
    "prompt", "width", "height", "duration", "seed", "unet_name", "clip_name",
    "vae_name", "audio_vae_name", "turbo", "turbo_lora", "turbo_strength",
    "turbo_steps",
]

_INPUT_NODE_ID = -10   # subgraph input routing pseudo-node
_OUTPUT_NODE_ID = -20  # subgraph output routing pseudo-node


def load_workflow(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_api_format(workflow: Dict[str, Any]) -> bool:
    if "nodes" in workflow and "links" in workflow:
        return False
    return any(isinstance(v, dict) and "class_type" in v for v in workflow.values())


def _norm_link(link: Any) -> Optional[Tuple]:
    """Normalize a link to (id, origin_id, origin_slot, target_id, target_slot)."""
    if isinstance(link, dict):
        return (
            link.get("id"),
            link.get("origin_id"),
            link.get("origin_slot"),
            link.get("target_id"),
            link.get("target_slot"),
        )
    if isinstance(link, list) and len(link) >= 5:
        return (link[0], link[1], link[2], link[3], link[4])
    return None


def set_subgraph_params(workflow: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Set friendly-named params on every subgraph instance's widgets_values."""
    subgraph_ids = {sg["id"] for sg in workflow.get("definitions", {}).get("subgraphs", [])}
    updated = 0
    for node in workflow.get("nodes", []):
        if node.get("type") not in subgraph_ids:
            continue
        widgets = node.setdefault("widgets_values", [])
        while len(widgets) < len(SUBGRAPH_WIDGET_ORDER):
            widgets.append(None)
        for name, value in params.items():
            if value is None or name not in SUBGRAPH_WIDGET_ORDER:
                continue
            widgets[SUBGRAPH_WIDGET_ORDER.index(name)] = value
        updated += 1
    return updated


# Widget index maps for the top-level (non-subgraph) MiniMax-H3 workflows such
# as the official R2V template, where the sampler graph is not in a subgraph.
_TOPLEVEL_WIDGETS = {
    "PrimitiveStringMultiline": {"prompt": 0},
    "PrimitiveFloat": {"duration": 0},
    "ResolutionSelector": {"megapixels": 1},
    "UNETLoader": {"unet_name": 0},
    "CLIPLoader": {"clip_name": 0},
    "VAELoader": {"vae_name": 0, "audio_vae_name": 0},
    "LoraLoaderModelOnly": {"turbo_lora": 0, "turbo_strength": 1},
}

# ResolutionSelector common (width, height, megapixels) presets for ~0.4 MP.
_RESOLUTION_PRESETS = {
    "16:9": ("16:9 (Widescreen)", 0.4),
    "9:16": ("9:16 (Portrait)", 0.4),
    "1:1": ("1:1 (Square)", 0.4),
    "4:3": ("4:3 (Classic)", 0.4),
    "3:4": ("3:4 (Portrait)", 0.4),
    "21:9": ("21:9 (Ultrawide)", 0.4),
}


def _set_widget(node: Dict[str, Any], index: int, value: Any) -> None:
    widgets = node.setdefault("widgets_values", [])
    while len(widgets) <= index:
        widgets.append(None)
    widgets[index] = value


def set_toplevel_params(workflow: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Inject params into a top-level (non-subgraph) MiniMax-H3 workflow.

    Used for the R2V template, whose sampler nodes are top-level rather than
    inside a subgraph instance. Returns the number of nodes updated.
    """
    updated = 0
    seen_vae = 0  # first VAELoader = video vae, second = audio vae
    for node in workflow.get("nodes", []):
        ntype = node.get("type")
        mapping = _TOPLEVEL_WIDGETS.get(ntype)
        if not mapping:
            continue
        changed = False
        for key, index in mapping.items():
            if ntype == "VAELoader":
                # Distinguish video vs audio VAE by order of appearance.
                key = "vae_name" if seen_vae == 0 else "audio_vae_name"
            if params.get(key) is None:
                continue
            _set_widget(node, index, params[key])
            changed = True
        if ntype == "ResolutionSelector" and params.get("ratio"):
            preset = _RESOLUTION_PRESETS.get(str(params["ratio"]))
            if preset:
                _set_widget(node, 0, preset[0])
                _set_widget(node, 1, preset[1])
                changed = True
        if ntype == "VAELoader":
            seen_vae += 1
        if changed:
            updated += 1
    return updated


def strip_unlinked_load_media(workflow: Dict[str, Any]) -> int:
    """Remove LoadImage/LoadVideo/LoadAudio nodes and their outbound links.

    The bundled R2V template ships reference images that are not part of the
    repo; a bare text prompt can't satisfy them. Dropping the nodes lets the
    workflow run in text-only mode (ref images/videos can be re-added by the
    caller via a raw workflow). Returns the number of links removed.
    """
    media_types = ("LoadImage", "LoadVideo", "LoadAudio")
    drop_ids = {n["id"] for n in workflow.get("nodes", []) if n.get("type") in media_types}
    if not drop_ids:
        return 0
    workflow["nodes"] = [n for n in workflow.get("nodes", []) if n["id"] not in drop_ids]
    links = workflow.get("links", [])
    kept = [l for l in links if _norm_link(l) and _norm_link(l)[1] not in drop_ids and _norm_link(l)[3] not in drop_ids]
    removed = len(links) - len(kept)
    workflow["links"] = kept
    return removed


def set_params(workflow: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Inject params regardless of whether the workflow uses a subgraph."""
    n = set_subgraph_params(workflow, params)
    if n == 0:
        set_toplevel_params(workflow, params)


def gui_to_api(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Return an API-format prompt dict for a GUI- or API-format workflow."""
    if is_api_format(workflow):
        return {k: v for k, v in workflow.items() if isinstance(v, dict) and "class_type" in v}
    try:  # pragma: no cover - depends on ComfyUI internals
        from execution import graph_to_prompt  # type: ignore

        prompt, _ = graph_to_prompt(copy.deepcopy(workflow))
        return prompt
    except Exception:
        pass
    return _manual_gui_to_api(workflow)


def _widget_value_map(node: Dict[str, Any]) -> Dict[str, Any]:
    """Map a node's widget values onto its unlinked input names positionally.

    Skips optional reference-media inputs (``ref_*``, ``image``, ``video``,
    ``audio``) that are unwired: leaving them unset keeps them optional, and
    positional widget values (dimensions etc.) must never be assigned to them.
    Empty-string widgets are also skipped.
    """
    widgets = node.get("widgets_values") or []
    named = [i for i in node.get("inputs", []) or [] if i.get("name")]
    unlinked = [i["name"] for i in named if i.get("link") is None]
    media_prefixes = ("ref_", "image", "video", "audio")
    out: Dict[str, Any] = {}
    for idx, name in enumerate(unlinked):
        if name.startswith(media_prefixes):
            continue
        if idx < len(widgets) and widgets[idx] not in (None, ""):
            out[name] = widgets[idx]
    return out


def _manual_gui_to_api(workflow: Dict[str, Any]) -> Dict[str, Any]:
    subgraphs = {sg["id"]: sg for sg in workflow.get("definitions", {}).get("subgraphs", [])}
    prompt: Dict[str, Any] = {}

    outer_src: Dict[int, Tuple] = {}
    for raw in workflow.get("links", []):
        l = _norm_link(raw)
        if l:
            outer_src[l[0]] = (str(l[1]), l[2])

    def add_plain(node: Dict[str, Any]) -> None:
        nid = str(node["id"])
        inputs: Dict[str, Any] = _widget_value_map(node)
        for inp in node.get("inputs", []) or []:
            if inp.get("link") is not None and inp["link"] in outer_src:
                o = outer_src[inp["link"]]
                inputs[inp["name"]] = [o[0], o[1]]
        # Nodes with no declared inputs (e.g. PrimitiveStringMultiline,
        # PrimitiveFloat, ResolutionSelector) still carry widget values that
        # downstream nodes consume; serialize the first widget under a
        # conventional key so it isn't dropped.
        if not inputs and node.get("widgets_values"):
            w = node["widgets_values"][0]
            if w not in (None, ""):
                inputs["value"] = w
        prompt[nid] = {"class_type": node.get("type"), "inputs": inputs}

    def expand_subgraph(node: Dict[str, Any]) -> str:
        """Expand a subgraph instance; return the inner VIDEO producer node id."""
        sg = subgraphs[node["type"]]
        prefix = f"{node['id']}:"
        inst_widgets = node.get("widgets_values") or []
        sg_inputs = sg.get("inputs", [])  # index == inputNode origin_slot

        inst_link: Dict[str, Any] = {}
        for inp in node.get("inputs", []) or []:
            if inp.get("link") is not None and inp["link"] in outer_src:
                inst_link[inp["name"]] = list(outer_src[inp["link"]])

        def widget_input_value(slot: int) -> Any:
            if slot >= len(sg_inputs):
                return None
            name = sg_inputs[slot].get("name")
            if name in inst_link:
                return inst_link[name]
            if name in SUBGRAPH_WIDGET_ORDER:
                idx = SUBGRAPH_WIDGET_ORDER.index(name)
                if idx < len(inst_widgets):
                    return inst_widgets[idx]
            return None

        inner_src: Dict[int, Tuple] = {}
        for raw in sg.get("links", []):
            l = _norm_link(raw)
            if l:
                inner_src[l[0]] = (l[1], l[2])

        def resolve_origin(origin_id: int, origin_slot: int) -> Any:
            if origin_id == _INPUT_NODE_ID:
                return widget_input_value(origin_slot)
            if origin_id == _OUTPUT_NODE_ID:
                return None
            return [f"{prefix}{origin_id}", origin_slot]

        for inner in sg.get("nodes", []):
            iid = inner.get("id")
            if iid in (_INPUT_NODE_ID, _OUTPUT_NODE_ID):
                continue
            nid = f"{prefix}{iid}"
            inputs: Dict[str, Any] = {}
            named = [i for i in inner.get("inputs", []) or [] if i.get("name")]
            for inp in inner.get("inputs", []) or []:
                link = inp.get("link")
                if link is not None and link in inner_src:
                    oid, oslot = inner_src[link]
                    val = resolve_origin(oid, oslot)
                    if val is not None:
                        inputs[inp["name"]] = val
            widgets = inner.get("widgets_values") or []
            unlinked = [i["name"] for i in named if i.get("link") is None]
            for idx, name in enumerate(unlinked):
                if idx < len(widgets) and widgets[idx] is not None and name not in inputs:
                    inputs[name] = widgets[idx]
            prompt[nid] = {"class_type": inner.get("type"), "inputs": inputs}

        # Determine the inner node that produces the subgraph's VIDEO output so
        # outer consumers (e.g. SaveVideo) can be redirected to it.
        producer: Optional[str] = None
        for out in sg.get("outputs", []) or []:
            for link_id in out.get("linkIds") or []:
                if link_id in inner_src:
                    oid, oslot = inner_src[link_id]
                    if oid not in (_INPUT_NODE_ID, _OUTPUT_NODE_ID):
                        producer = f"{prefix}{oid}"
        return producer or f"{prefix}"

    # Expand/flatten all nodes; remember each subgraph instance's VIDEO producer.
    sg_video_producer: Dict[str, str] = {}
    for node in workflow.get("nodes", []):
        if node.get("type") in subgraphs:
            sg_video_producer[str(node["id"])] = expand_subgraph(node)
        else:
            add_plain(node)

    # Redirect outer inputs that referenced a subgraph instance to its inner
    # VIDEO producer node.
    for node in prompt.values():
        for iname, val in list(node.get("inputs", {}).items()):
            if isinstance(val, list) and len(val) == 2 and val[0] in sg_video_producer:
                node["inputs"][iname] = [sg_video_producer[val[0]], val[1]]

    return prompt


def set_prompt_text(prompt: Dict[str, Any], text: str) -> bool:
    """Inject the prompt into the MiniMaxH3 sampler's prompt input (post-flatten)."""
    for node in prompt.values():
        if "MiniMaxH3" in str(node.get("class_type", "")):
            node.get("inputs", {})["prompt"] = text
            return True
    for node in prompt.values():
        ct = str(node.get("class_type", ""))
        if "TextEncode" in ct or "CLIPTextEncode" in ct:
            inputs = node.get("inputs", {})
            for key in ("text", "prompt"):
                if key in inputs:
                    inputs[key] = text
                    return True
    return False
