# Graph Evidence Workbench — Build Contract (v0.1)

Shared interface for the agent team building `graph-evidence-workbench.html`.
**Every module builds against this. Do not invent new global names, DOM IDs, data
fields, or event names — if something is missing, flag it, don't improvise.**

Target page: `Profiling_Insight_and_Tool/AI_Profiling_Tool/graph-evidence-workbench.html`
Light mode only: `<html data-theme="light">`. Consume PTO tokens/patterns; never
re-implement pattern geometry or invent new visual styles.

Vendor paths (relative from the page):
- `../../vendor/pto-design-system/tokens/{foundation,semantic,components}.css`
- `../../vendor/pto-design-system/css/style.css`
- `../../vendor/pto-design-system/patterns/model-graphviz/pattern.{css,js}`  → `window.PtoModelGraphvizPattern`
- `../../vendor/pto-design-system/patterns/swimlane-task/pattern.{css,js}`    → `window.PtoSwimlaneTaskPattern`

---

## 1. Global namespace

Everything hangs off `window.GEW` (defined in `core.js`, loaded first).

```js
window.GEW = {
  SCHEMA_VERSION: '0.1',
  state: { reportId, selectedNodeId, activeFilter, selectedStepId, data },
  bus,            // event bus, see §2
  util,           // helpers, see §6
  // module singletons attached by their files:
  loader, traceParser, graphStage, inspector, swimlane, app
};
```

`GEW.state.data` (set by loader after load):
`{ graph, nodeInfo, problemMap, report, laneModel }`

## 2. Event bus (`GEW.bus`)

```js
GEW.bus.on(type, fn)       // returns unsubscribe()
GEW.bus.emit(type, detail) // detail is a plain object
```

Event catalog (the ONLY events; detail shapes are fixed):

| type | detail | emitted by | consumed by |
|---|---|---|---|
| `report:loaded`   | `{ reportId, data }`            | app    | all stages |
| `selection:change`| `{ nodeId, source }`            | graph / swimlane / inspector / app | all |
| `filter:change`   | `{ priority }`                  | header / app | graphStage, inspector |
| `step:focus`      | `{ stepId }`                    | swimlane / app | swimlane |

`source` ∈ `'graph' | 'swimlane' | 'inspector' | 'mapped' | 'url' | 'init'`.
`priority` ∈ `'all' | 'P0' | 'P1' | 'P2' | 'off'`.
Modules must ignore an event whose `source` is themselves (avoid feedback loops).

## 3. DOM IDs (provided by the HTML shell — do not rename)

```
#gew-root
  #gew-header
    #gew-home                 (Home button)
    #gew-report-name          (text: report title)
    #gew-source               (data-source status text)
    #gew-filter               (segment-control; buttons have data-priority=all|P0|P1|P2|off)
    #gew-export               (button: export JSON snapshot)
    #gew-copy                 (button: copy evidence summary)
  #gew-main
    #gew-graph-shell
      #gew-graph-stage        (model-graphviz render target)
      #gew-graph-empty        (empty-state for non-Qwen / no graph; hidden by default)
      #gew-graph-toolbar      (fit / zoom controls)
    #gew-inspector            (panel-shell)
      #gew-inspector-body
  #gew-swimlane
    #gew-swimlane-tabs        (segment-control; data-lane=step|stream|communication|overlap|coverage)
    #gew-swimlane-body        (canvas + axis live here)
    #gew-swimlane-empty       (data-gap empty state; hidden by default)
```

## 4. Data schemas (files under `AI_Profiling_Tool/data/`)

All non-trace JSON files carry these top-level fields:
`schemaVersion` (== '0.1'), `modelId` ('qwen2-7b'), `reportId`, `sourceFiles` (string[]), `generatedAt` (ISO).

### `qwen2-7b.graph.json`
```jsonc
{ "schemaVersion":"0.1","modelId":"qwen2-7b","reportId":"...","sourceFiles":[...],"generatedAt":"...",
  "graph": {
    "width": 1280, "height": 1540,
    "clusters": [ { "id","label","x","y","width","height","colorKey"? } ],
    "clusterChildren": { "<clusterId>": ["<childClusterId|nodeId>", ...] },   // for collapse
    "nodes":    [ { "id","label","typeLabel","kind"('tensor'|'op'|'module'),"x","y","width","height","colorKey","parent"? } ],
    "edges":    [ { "source","target","color"?,"dashed"? } ]
  } }
```

### `qwen2-7b.node-info.json`
```jsonc
{ ...meta, "nodeInfo": { "<nodeId>": {
    "what","idEn","clusters":[..],"inputs":[{"from","desc"}],"outputs":[{"to","desc"}],"params"?,"sources":[{"text","url"}]
} } }
```

### `qwen2-7b.problem-map.json`
```jsonc
{ ...meta, "defaultNode":"lm_head",
  "problemNodes": { "<nodeId>": {
    "priority":"P0|P1|P2","issueId":"...","issueRef":"3.2","opType":"MatMulV3 ...",
    "title","metric","impact","fix":[..],"verify","codeHint",
    "runtimeOpName"?,"streamId"?,"stepId"?         // optional linkage keys into the trace
} } }
```

### `qwen2-7b.demo-report.json`  (inspector content)
```jsonc
{ ...meta, "issues": { "<nodeId>": {
    "diagnosis": { "priority","dimension","title","summary","metrics":[["label","value"],...] },
    "evidence":  [ { "text","sourceFile","sourceField"?,"confidence":"raw|derived|inferred" } ],
    "operators": [ { "name","runtimeOpName"?,"value","streamId"? } ],
    "actions":   [ { "text","evidenceRef"?,"inferred"? } ],
    "verification":[ { "metric","current","target" } ],   // never "重采验证" with no numbers
    "mappedNodes":[ "<nodeId>", ... ],
    "coverage":  { "covered":["<nodeId>",..],"missing":["<dataFile/field>",..],"ratio":0.0 }
} } }
```

### `qwen2-7b.demo.trace_view.json`  (Chrome Trace Event Format; parser input)
Either wrapped or raw is accepted by the parser:
```jsonc
{ ...meta?, "traceEvents": [
    { "ph":"M","name":"process_name","pid":1,"args":{"name":"python_pid"} },
    { "ph":"M","name":"thread_name","pid":1,"tid":7,"args":{"name":"Ascend Hardware/Stream 47"} },
    { "ph":"X","name":"MatMulV3","pid":1,"tid":7,"ts":1773295061394254.8,"dur":958.1,
      "args":{ "Step Id":11,"Stream Id":47,"Task Id":..,"Wait Time(us)":1157.6,"Input Shapes":"..","Connection Id":.. } }
] }
```
Layering by process/thread name: `python` / `CANN` (AscendCL,GE,Runtime) / `Ascend Hardware` (Stream N) /
`Communication` (HCCL/hcom). Step boundaries from `step trace`/`Step Id`. Demo MUST be parsed by the
real parser (no hand-written final lane bars). Keep ~a few hundred events, realistic.

### `qwen2-7b.demo.evidence.json`  (regression fixture only — NOT a render source)
Expected `laneModel` summary the parser should produce from the demo trace (counts, step ids, stream ids).

## 5. Parser → swimlane contract (`GEW.LaneModel`)

`GEW.traceParser.parse(traceJson, { problemMap }) -> LaneModel`:
```js
LaneModel = {
  timeRange: { startUs, endUs },
  steps: [ { stepId, startUs, durUs, compute, comm, free, overlap } ],   // step lane bars
  lanes: [ {
    id, kind:'step'|'stream'|'communication'|'overlap'|'coverage', label,
    tasks: [ {
      id, label, opName, runtimeOpName?, nodeId?,        // nodeId links back to graph
      startUs, durUs, waitUs?, streamId?, stepId?, rankId?,
      status?:'wait'|'overlap'|'ok', sourceFile?, laneKind   // laneKind feeds the pattern
    } ]
  } ],
  byNode: { '<nodeId>': ['<taskId>', ...] },   // graph→swimlane linkage index
  gaps:  [ { laneId, kind, reason } ]          // missing-data markers → empty/partial lane
}
```
Swimlane maps each task to `PtoSwimlaneTaskPattern.drawTaskBar(ctx,{task,x,y,width,height,baseColor,isSelected,isRelated,...})`.
Pattern reads `opName/laneKind/totalCycle/clcCycle/gap/status/...` — map `durUs→totalCycle`, `waitUs→gap`,
`(durUs-waitUs)→clcCycle`, keep `opName`,`laneKind`,`status`. Tooltip via `initHoverTooltip`/`formatTaskTooltip`;
ensure tooltip has op/lane/start/duration/wait/stream/rank/source.

## 6. Module APIs (attach to `GEW`)

```js
GEW.loader.load(reportId) -> Promise<data>   // fetch all JSON, validate SCHEMA_VERSION, parse trace → laneModel; throws GEW.SchemaError on mismatch
GEW.traceParser.parse(traceJson, opts) -> LaneModel

GEW.graphStage.init({ container, graph, problemMap })   // render via PtoModelGraphvizPattern, wire pan/zoom/cluster/click
GEW.graphStage.select(nodeId)                            // highlight + dim others (no event)
GEW.graphStage.focusNode(nodeId)                         // pan/zoom to node
GEW.graphStage.setFilter(priority)                       // show only matching overlays
GEW.graphStage.fit()

GEW.inspector.init({ container })
GEW.inspector.render(nodeId)                             // pulls from GEW.state.data
GEW.inspector.clear()                                    // 'off' / empty state

GEW.swimlane.init({ container, laneModel })
GEW.swimlane.highlightNode(nodeId)                       // scroll/emphasize related tasks
GEW.swimlane.focusStep(stepId)
GEW.swimlane.setTab(laneKind)

GEW.app.start()                                          // boot: read URL params, load, init stages, wire bus
```

`GEW.util`: `qs(id)`, `el(tag,attrs,children)`, `fmtUs(n)`, `escapeHtml(s)`,
`getParams()` ({reportId,nodeId,priority,stepId}), `setParams(obj)`.

## 7. Linkage rules (bidirectional)
- graph node click → `selection:change{source:'graph'}` → inspector.render + swimlane.highlightNode.
- swimlane task click → `selection:change{source:'swimlane'}` → graph.select + graph.focusNode + inspector.render.
- inspector Mapped-Node click → `selection:change{source:'mapped'}` → graph.focusNode + swimlane.highlightNode.
- header filter → `filter:change` → graph.setFilter + inspector (mapped-list reflects filter).
- non-Qwen / no graph → show `#gew-graph-empty`, hide stages.
- missing trace → `#gew-swimlane-empty` with gap reasons; never fake full lanes.

## 8. Forbidden
- No CDN runtime deps. No new buttons/badges/cards outside PTO classes.
- No re-implementing model-graphviz node geometry or swimlane task-bar segments in page CSS/DOM.
- No `border-left` accent rails; full 1px border + bg for callouts. No card-in-card nesting.
- No hardcoded business data in HTML/JS — everything from `data/*.json`.
