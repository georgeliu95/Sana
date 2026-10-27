"""Self-contained HTML rendering; embedded evidence needs no external services."""

from __future__ import annotations

from typing import Final

_TEMPLATE: Final = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3Adopted · matched209 对比与 Nsys explorer</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182332;background:#f4f6fa}body{margin:0;padding:26px;max-width:1900px;margin:auto}h1{font-size:25px;margin:0 0 8px}h2{font-size:18px;margin:24px 0 10px}p{line-height:1.65;margin:9px 0}.card{background:white;border:1px solid #d8e0eb;border-radius:10px;padding:17px;margin:14px 0}.warning{border-left:5px solid #bd5510;background:#fff7ed}.muted{color:#566579;font-size:13px}.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}select,input,button{padding:8px 10px;border:1px solid #cbd5e1;border-radius:6px;background:white;color:inherit}input{min-width:240px;flex:1}button{cursor:pointer}button.active{background:#12396b;color:white}.scroll{overflow:auto;max-height:760px}table{width:100%;border-collapse:collapse;font-size:12px}th{background:#eaf0f8;position:sticky;top:0;text-align:left;white-space:nowrap;z-index:1}th,td{padding:8px;border-bottom:1px solid #e4eaf2;vertical-align:top}td{font-variant-numeric:tabular-nums}td.symbol{min-width:330px;max-width:650px;overflow-wrap:anywhere;font-family:ui-monospace,monospace;font-size:11px}td.path{max-width:420px;overflow-wrap:anywhere}tr.selectable:hover{background:#eef5ff;cursor:pointer}.unknown{background:#fff6de}a{color:#155aa7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.total{font-size:14px;font-weight:600;padding:10px 0}.pill{padding:3px 8px;border-radius:5px;background:#e9eff7;display:inline-block;margin-right:6px}.links{display:flex;gap:16px;flex-wrap:wrap}.bar{height:4px;background:#2e6caf;max-width:150px;margin-top:4px}.empty{padding:20px;color:#68768a}
</style>
<body><h1>H3Adopted：896×512 · 209 帧 · 24 FPS</h1>
<p class="muted">sparkcluster-node3 / 单 GB10 SM121 / seed 0。端到端范围：fresh Qwen 请求进入 → 最终音视频 MP4 完成。</p>
<div class="card warning"><strong>学生模型、量化精度和 decoder 均已改变。</strong>
<p>投影来源：pruned scalar-FP8 represented → BF16（scale 亦 BF16）→ 完整 student LoRA → MXFP8；四步 FastH3 + TAEH3。历史三个分母各为 n=1 的 Nsys 采集，不能当作本次 fresh A/B 或同模型消融。</p>
<p>新“0 cache hits”只指未安装 FirstBlockCache / block-stack skip。请求内部仍保留 RoPE、text-refiner 缓存和七行 AdaLN lookup。视频 8.708333 秒，原 PCM 8.700 秒，约 8.33 毫秒尾差；保留原始音频。</p></div>
<div class="card"><h2>端到端对比</h2><p id="lead"></p><div id="comparison" class="scroll"></div>
<p class="muted">加速倍数 = 历史单次端到端 / 新普通请求均值，仅作跨时间、跨模型路线参考。n=1 无法估计请求级方差；新额外 profile 时间单列。</p>
<div class="links"><a href="comparison.csv">comparison.csv</a><a href="modules.csv">modules.csv</a><a href="operators.csv">operators.csv</a><a href="kernels.csv">kernels.csv</a><a href="summary.json">全部证据 summary.json</a></div></div>
<div class="card"><h2>新请求明细</h2><div id="samples" class="scroll"></div>
<p class="muted">decode_and_mux_s 已包含在 native_generate_s 内且不含 audio VAE；这些区间不能相加。启动与预热另列并排除于热请求时间。</p><p id="warmup" class="muted"></p></div>
<div class="card"><h2>Nsys module → operator → kernel</h2><p id="profileStatus"></p>
<p class="muted">GPU秒为 kernel duration 求和，排除 memcpy/memset，不是 SM utilization。归属依据按行标明：CUDA launch → NVTX，或官方 NVTX GPU-projected phase window；一个 kernel 只计一次。Kernel 分位数保留 symbol/grid/block/path 分桶，不能合并相加。</p>
<div class="filters"><select id="route"><option value="">全部路线</option></select><select id="module"><option value="">全部 module</option></select><select id="operator"><option value="">全部 operator</option></select><input id="query" placeholder="检索 symbol、NVTX path 或优化项"><select id="sort"><option value="total_gpu_seconds">累计 GPU 秒 ↓</option><option value="kernel_count">次数 ↓</option><option value="p95_ms">P95 ↓</option><option value="mean_ms">均值 ↓</option></select><select id="limit"><option value="200">前 200 行</option><option value="1000">前 1000 行</option><option value="0">全部行</option></select><button id="reset">重置</button></div>
<p><button data-tab="modules" class="active">Module</button> <button data-tab="operators">Operator family</button> <button data-tab="kernels">Kernel / shape / path</button> <button data-tab="inventory">原优化盘点</button></p>
<div id="totals" class="total"></div><div id="explorer" class="scroll"></div>
<p class="muted">点击 module 或 operator 行可下钻；完整 kernel symbol 保留。Unknown 与历史缺失的 shape/分位数保持可见，不填推断值。原 inventory 是事前分析，不能据此声称101项全部启用。</p></div>
<div class="card"><h2>FBC / NFE 口径</h2><p>Fullopt：49 次 forward = 16 次完整 50-block stack + 33 次缓存部分更新，block 0 仍执行49次。Cache-management GPU <strong>0.962488714秒</strong>已包含在 Fullopt 总 GPU 时间内。其 CPU stream 等待包含已计GPU工作，无额外E2E开销消融。</p><p>新路线：每请求实测4次完整stack，200次主干block evaluation；FBC命中0。管理GPU=0只表示没有FBC管理kernel，不代表其他内存管理耗时为0。</p></div>
<details class="card"><summary>统计方法与边界</summary><ul id="methods"></ul></details>
<details class="card"><summary>来源、SHA256、命令与过滤 scope</summary><pre id="evidence"></pre></details>
<details class="card"><summary>新 NVTX / CUDA Runtime / memcpy 原始证据（非相加模块）</summary><pre id="auxiliary"></pre></details>
<script id="payload" type="application/json">__REPORT_JSON__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent),$=id=>document.getElementById(id);
const label={route:"路线",provenance:"统计来源",timing_samples:"普通 n",profile_samples:"profile n",e2e_mean_s:"E2E均值(s)",e2e_median_s:"E2E中位(s)",e2e_min_s:"最小(s)",e2e_max_s:"最大(s)",e2e_sample_stddev_s:"样本SD(s)",profiled_e2e_s:"额外profile E2E(s)",gpu_kernel_seconds:"GPU kernel总秒",kernel_count:"kernel次数",h3_forward_calls:"H3 forward",h3_full_stack_nfe:"完整stack NFE",h3_cache_partial_hits:"FBC命中",cache_management_gpu_seconds:"FBC管理GPU(s，已含)",speedup_vs_historical_baseline:"相对历史Baseline",speedup_vs_historical_fullopt:"相对历史Fullopt",speedup_vs_historical_spark:"相对历史Spark",module:"module",operator_family:"operator family",operator_evidence:"归类依据",kernel_name:"完整 kernel symbol",total_gpu_seconds:"GPU总秒",gpu_seconds:"GPU总秒",mean_ms:"均值(ms)",p50_ms:"P50(ms)",p95_ms:"P95(ms)",p99_ms:"P99(ms)",min_ms:"min(ms)",max_ms:"max(ms)",nvtx_path:"归属 path（可能为投影派生）",original_nvtx_path:"原始 launch NVTX path",original_source_key:"原始 bucket key",attribution_method:"归属方法",bucket_attribution:"分桶证据",capture_origin:"采集口径",source_key:"来源bucket key",profiled:"是否额外profile",request_id:"request",e2e_s:"E2E(s)",qwen_s:"Qwen请求(s)",native_generate_s:"native_generate(s)",decode_and_mux_s:"decode+MUX(s，子区间)",actual_nfe:"完整stack NFE",cache_hits:"FBC命中"};
const records=t=>t.rows.map(row=>Object.fromEntries(t.columns.map((key,i)=>[key,row[i]])));
const tables={modules:D.profile_tables.modules,operators:D.profile_tables.operators,kernels:D.profile_tables.kernels,inventory:D.inventory};
let tab="modules";
const format=(value,key)=>{if(value===""||value===null||value===undefined)return"—";if(typeof value==="object")return JSON.stringify(value);if(/(_s$|seconds$|_ms$|^speedup)/.test(key)&&Number.isFinite(Number(value)))return Number(value).toFixed(key.startsWith("speedup")?2:6);return String(value)};
function draw(target,columns,rows,onClick){const fragment=document.createDocumentFragment(),table=document.createElement("table"),head=table.createTHead().insertRow();for(const key of columns){const th=document.createElement("th");th.textContent=label[key]||key;head.append(th)}const body=table.createTBody();for(const row of rows){const tr=body.insertRow();if(onClick){tr.className="selectable";tr.addEventListener("click",()=>onClick(row))}if(row.module==="Unknown"||row.operator_family==="Unknown"||row.module==="Other / unassigned")tr.classList.add("unknown");for(const key of columns){const td=tr.insertCell();td.textContent=format(row[key],key);td.title=String(row[key]??"缺失，未推断");if(key==="kernel_name")td.className="symbol";if(key==="nvtx_path")td.className="path";if(key==="gpu_seconds"&&Number.isFinite(Number(row.share_of_route_kernel_time_percent))){const bar=document.createElement("div");bar.className="bar";bar.style.width=Math.min(100,Number(row.share_of_route_kernel_time_percent))+"%";td.append(bar)}}}fragment.append(table);if(!rows.length){const empty=document.createElement("p");empty.className="empty";empty.textContent="此筛选下没有已采集行。";fragment.append(empty)}target.replaceChildren(fragment)}
const comparison=records(D.comparison),fresh=comparison.find(r=>r.route==="H3Adopted");
$("lead").textContent="新普通请求 n="+fresh.timing_samples+"，均值 "+format(fresh.e2e_mean_s,"e2e_mean_s")+" 秒；最小 "+format(fresh.e2e_min_s,"e2e_min_s")+"、最大 "+format(fresh.e2e_max_s,"e2e_max_s")+" 秒。";
draw($("comparison"),["route","provenance","timing_samples","profile_samples","e2e_mean_s","e2e_sample_stddev_s","e2e_min_s","e2e_max_s","profiled_e2e_s","gpu_kernel_seconds","kernel_count","h3_forward_calls","h3_full_stack_nfe","h3_cache_partial_hits","cache_management_gpu_seconds","speedup_vs_historical_baseline","speedup_vs_historical_fullopt","speedup_vs_historical_spark"],comparison);
const requests=D.new_results.values.requests;
draw($("samples"),["request_id","profiled","e2e_s","qwen_s","native_generate_s","decode_and_mux_s","actual_nfe","cache_hits","output"],requests);
$("warmup").textContent="启动和预热总耗时："+format(D.new_results.values.startup_and_warmup_s,"startup_and_warmup_s")+" 秒（不计入以上普通请求）。";
$("profileStatus").textContent=D.profile_ready?"新profile数据已采集；以下新GPU表仅来自额外profile请求。":"新profile待采集或导出不完整；新GPU字段留空，不能解释为0。";
for(const method of D.methodology){const li=document.createElement("li");li.textContent=method;$("methods").append(li)}
$("evidence").textContent=JSON.stringify(D.evidence,null,2);$("auxiliary").textContent=JSON.stringify(D.auxiliary_profiles,null,2);
function options(id,values){const select=$(id),current=select.value;while(select.options.length>1)select.remove(1);for(const value of [...new Set(values)].sort()){const option=new Option(value,value);select.add(option)}select.value=current}
options("route",comparison.map(r=>r.route));options("module",records(tables.kernels).map(r=>r.module));options("operator",records(tables.kernels).map(r=>r.operator_family));
function update(){let rows=records(tables[tab]);const route=$("route").value,module=$("module").value,operator=$("operator").value,query=$("query").value.toLocaleLowerCase();rows=rows.filter(r=>(!route||tab==="inventory"||r.route===route)&&(!module||tab==="inventory"||r.module===module)&&(!operator||tab==="inventory"||tab==="modules"||r.operator_family===operator)&&(!query||Object.values(r).join(" ").toLocaleLowerCase().includes(query)));let sort=$("sort").value;if(sort==="total_gpu_seconds"&&tab!=="kernels")sort="gpu_seconds";rows.sort((a,b)=>(Number(b[sort])||0)-(Number(a[sort])||0));const count=rows.reduce((a,r)=>a+(Number(r.kernel_count)||0),0),seconds=rows.reduce((a,r)=>a+(Number(r.total_gpu_seconds??r.gpu_seconds)||0),0),limit=Number($("limit").value);$("totals").textContent=(tab==="inventory"?rows.length+" 个事前优化/实验条目":rows.length+" 个分桶 / "+count+" 次kernel / "+seconds.toFixed(6)+" GPU秒")+(limit&&rows.length>limit?"（展示前"+limit+"行）":"");const displayed=limit?rows.slice(0,limit):rows;draw($("explorer"),tables[tab].columns,displayed,["modules","operators"].includes(tab)?r=>{$("route").value=r.route;$("module").value=r.module;if(r.operator_family)$("operator").value=r.operator_family;activate("kernels")}:null)}
function activate(name){tab=name;$("operator").disabled=tab==="modules"||tab==="inventory";$("module").disabled=tab==="inventory";for(const button of document.querySelectorAll("[data-tab]"))button.classList.toggle("active",button.dataset.tab===tab);update()}
for(const button of document.querySelectorAll("[data-tab]"))button.addEventListener("click",()=>activate(button.dataset.tab));
for(const id of ["route","module","operator","query","sort","limit"])$(id).addEventListener(id==="query"?"input":"change",update);
$("reset").addEventListener("click",()=>{for(const id of ["route","module","operator","query"])$(id).value="";activate("modules")});update();
</script></body></html>"""


def render_html(serialized_report: str) -> str:
    """Embed JSON safely against script-ending strings in prompts or symbols."""
    escaped = (
        serialized_report.replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
    )
    return _TEMPLATE.replace("__REPORT_JSON__", escaped)
